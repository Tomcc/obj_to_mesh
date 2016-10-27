extern crate wavefront_obj;
extern crate byteorder;
extern crate clap;
extern crate half;

use clap::{Arg, App};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::prelude::*;
use std::fs::File;
use wavefront_obj::obj::{ObjSet, Object, Shape, VTNIndex, Vertex, TVertex, Normal};
use std::mem::size_of;
use std::collections::HashMap;
use std::f64;
use std::path::Path;
use half::f16;

fn pack_normalized(val: f64, max: u32) -> u32 {
	f64::ceil(val * max as f64) as u32
}

fn pack_i2_10_10_10(normal: Normal) -> u32 {
	(pack_normalized(normal.x, 511) << 0) |
	(pack_normalized(normal.y, 511) << 10) |
	(pack_normalized(normal.z, 511) << 20)
}

fn pack_f16(val: f64) -> u16 {
	//attempt to fix bad exports
	let mut x = val;
	while x > 1. {
		x -= 1.;
	}
	while x < -1. {
		x += 1.;
	}
	f16::from_f64(x).as_bits()
}

#[derive(Clone, Copy)]
enum Attribute {
	Position,
	Normal,
	Tangent,
	Tex0,
}

fn size_of_attribute(attr: Attribute) -> usize {
	match attr {
		Attribute::Position => size_of::<f32>() * 3,
		Attribute::Normal => size_of::<u32>(),
		Attribute::Tangent => size_of::<u32>(),
		Attribute::Tex0 => size_of::<f16>() * 2,
	}
}

#[derive(Debug, Clone, Copy)]
struct VertexFieldOffsets {
	normal: Option<usize>,
	tangent: Option<usize>,
	tex0: Option<usize>,
}

fn has_attribute(vtni: VTNIndex, attr: Attribute) -> bool {
	let (_, tex, normal) = vtni;
	match attr {
		Attribute::Position => true,
		Attribute::Normal => normal.is_some(),
		Attribute::Tex0 => tex.is_some(),
		_ => false,
	}
}

fn has_all(obj: &Object, attr: Attribute) -> bool {

	for geo in &obj.geometry {
		for shape in &geo.shapes {
			match *shape {
				Shape::Triangle(v1, v2, v3) => {
					if !has_attribute(v1, attr) || !has_attribute(v2, attr) || !has_attribute(v3, attr) {
						return false;
					}
				},
				_=> panic!("Unsupported primitive mode")
			}
		}
	}
	true
}

fn get_offset(obj: &Object, attr: Attribute, offset: &mut usize) -> Option<usize> {
	let orig_offs = *offset;
	if has_all(obj, attr) {
		*offset += size_of_attribute(attr);
		return Some(orig_offs);
	}
	None
}

impl VertexFieldOffsets {
	fn from_object(obj: &Object, with_tangent: bool) -> Self {
		let mut offset = size_of_attribute(Attribute::Position);

		VertexFieldOffsets {
			normal: get_offset(obj, Attribute::Normal, &mut offset),
			tangent: 
				if with_tangent {
					let orig_offset = offset;
					offset += size_of_attribute(Attribute::Tangent);
					Some(orig_offset)
				}
				else {
					None
				},
			tex0: get_offset(obj, Attribute::Tex0, &mut offset),
		}
	}
}

#[derive(Clone, Debug)]
struct GPUVertex {
	pos: Vertex,
	normal: Option<Vertex>,
	tangent: Option<Vertex>,
	tex: Option<TVertex>,
}

impl GPUVertex {
	fn from_vtni_and_obj(vtni: VTNIndex, obj: &Object, format: &VertexFieldOffsets) -> Self {
		let (pos_idx, tex_opt_idx, norm_opt_idx) = vtni;
		GPUVertex {
			pos: obj.vertices[pos_idx],
			normal: match norm_opt_idx {
			    Some(idx) if format.normal.is_some() => Some(obj.normals[idx]),
			    _ => None,
			},
			tangent: None,
			tex: match tex_opt_idx {
			    Some(idx) if format.tex0.is_some() => Some(obj.tex_vertices[idx]),
			    _ => None,
			},
		}
	}

	fn write_to(&self, data: &mut Vec<u8>) {
		data.write_f32::<LittleEndian>(self.pos.x as f32).unwrap();
		data.write_f32::<LittleEndian>(self.pos.y as f32).unwrap();
		data.write_f32::<LittleEndian>(self.pos.z as f32).unwrap();

		if let Some(normal) = self.normal {
			data.write_u32::<LittleEndian>(pack_i2_10_10_10(normal)).unwrap();
		}

		if let Some(tangent) = self.tangent {
			data.write_u32::<LittleEndian>(pack_i2_10_10_10(tangent)).unwrap();
		}

		if let Some(tex) = self.tex {
			data.write_u16::<LittleEndian>(pack_f16(tex.x)).unwrap();
			data.write_u16::<LittleEndian>(pack_f16(tex.y)).unwrap();
		}
	}
}

#[derive(Debug)]
struct Mesh {
	vertices: Vec<GPUVertex>,
	indices: Vec<usize>,
	map: HashMap<VTNIndex, usize>,
	format: VertexFieldOffsets,

	min: Vertex,
	max: Vertex,
}

fn flt_min(a: f64, b: f64) -> f64 {
	if a < b { a } else { b }
}

fn flt_max(a: f64, b: f64) -> f64 {
	if a > b { a } else { b }
}

fn vert_min(a: Vertex, b: Vertex) -> Vertex {
	Vertex {
		x: flt_min(a.x, b.x),
		y: flt_min(a.y, b.y),
		z: flt_min(a.z, b.z),
	}
} 

fn vert_max(a: Vertex, b: Vertex) -> Vertex {
	Vertex {
		x: flt_max(a.x, b.x),
		y: flt_max(a.y, b.y),
		z: flt_max(a.z, b.z),
	}
} 

fn addmut(dst: &mut Vertex, src: Vertex) {
	dst.x += src.x;
	dst.y += src.y;
	dst.z += src.z;
}

fn lenght(v: Vertex) -> f64 {
	f64::sqrt( v.x * v.x + v.y * v.y + v.z * v.z )
}
fn normalize(v: Vertex) -> Vertex {
	mul(v, 1.0 / lenght(v))
}

fn dot(a: Vertex, b: Vertex) -> f64 {
	a.x * b.x + a.y * b.y + a.z * b.z
}

fn sub(a: Vertex, b: Vertex) -> Vertex {
	Vertex{
		x: a.x - b.x,
		y: a.y - b.y,
		z: a.z - b.z,
	}
}

fn mul(a: Vertex, b: f64) -> Vertex {
	Vertex{
		x: a.x * b,
		y: a.y * b,
		z: a.z * b,
	}
}

impl Mesh {
	fn from_object(obj: &Object, generate_tangents: bool) -> Self {
		let format = VertexFieldOffsets::from_object(&obj, generate_tangents);
		let mut mesh = Mesh {
			vertices: Vec::new(),
			indices: Vec::new(),
			map:HashMap::new(),
			min: Vertex{x: f64::MAX, y: f64::MAX, z: f64::MAX },
			max: Vertex{x: f64::MIN, y: f64::MIN, z: f64::MIN },
			format: format,
		};

		for geo in &obj.geometry {
			for shape in &geo.shapes {
				match *shape {
					Shape::Triangle(v1, v2, v3) => {
						mesh.add_index(v1, &obj, &format);
						mesh.add_index(v2, &obj, &format);
						mesh.add_index(v3, &obj, &format);
					},
					_=> panic!("Unsupported primitive mode")
				}
			}
		}

		if generate_tangents {
			//http://gamedev.stackexchange.com/questions/68612/how-to-compute-tangent-and-bitangent-vectors

			let mut tan1 = vec!(Vertex{x: 0.0, y: 0.0, z:0.0}; mesh.vertices.len());

			let mut ii = 0;
			while ii < mesh.indices.len() {
				let i1 = mesh.indices[ii + 0];
				let i2 = mesh.indices[ii + 1];
				let i3 = mesh.indices[ii + 2];

				let v1 = mesh.vertices[i1].pos;
				let v2 = mesh.vertices[i2].pos;
				let v3 = mesh.vertices[i3].pos;

				let w1 = mesh.vertices[i1].tex.unwrap();
				let w2 = mesh.vertices[i2].tex.unwrap();
				let w3 = mesh.vertices[i3].tex.unwrap();

				let x1 = v2.x - v1.x;
				let x2 = v3.x - v1.x;
				let y1 = v2.y - v1.y;
				let y2 = v3.y - v1.y;
				let z1 = v2.z - v1.z;
				let z2 = v3.z - v1.z;

				let s1 = w2.x - w1.x;
				let s2 = w3.x - w1.x;
				let t1 = w2.y - w1.y;
				let t2 = w3.y - w1.y;

				let r = 1.0 / (s1 * t2 - s2 * t1);
				let sdir = Vertex{
					x: (t2 * x1 - t1 * x2) * r, 
					y: (t2 * y1 - t1 * y2) * r,
					z: (t2 * z1 - t1 * z2) * r,
				};
				
				addmut(&mut tan1[i1], sdir);
				addmut(&mut tan1[i2], sdir);
				addmut(&mut tan1[i3], sdir);
				
				ii += 3;
			}

			for a in 0..mesh.vertices.len()	{
				let n = mesh.vertices[a].normal.unwrap();
				let t = tan1[a];

				// Gram-Schmidt orthogonalize
				mesh.vertices[a].tangent = Some(normalize(sub(t,mul(n, dot(n, t)))));
			}
		}

		mesh
	}	

	fn create_vertex(&mut self, vtni: VTNIndex, obj: &Object, format: &VertexFieldOffsets) -> usize {
		let idx = self.vertices.len();

		let v = GPUVertex::from_vtni_and_obj(vtni, obj, format);

		self.min = vert_min(self.min, v.pos);
		self.max = vert_max(self.max, v.pos);

		self.vertices.push( v );

		idx
	}

	fn add_index(&mut self, vtni: VTNIndex, obj: &Object, format: &VertexFieldOffsets) {
		if let Some(idx) = self.map.get(&vtni) {
			self.indices.push(*idx);
			return;
		}

		let idx = self.create_vertex(vtni, obj, format);
		self.map.insert(vtni, idx);
		self.indices.push(idx);
	}

	fn get_index_size(&self) -> usize {
		match self.vertices.len() {
		    n if n <= 0xff => 1,
		    n if n <= 0xffff => 2,
		    _ => 4
		}
	}
}

fn convert_obj(obj: Object, generate_tangents: bool) -> Vec<u8> {

	//build a VTNIndex => Vertex map and build actual vertices
	let mesh = Mesh::from_object(&obj, generate_tangents);

	let mut data = vec![];

	//write the index size in bytes
	let index_size = mesh.get_index_size() as u8;
	data.write_u8(index_size).unwrap();

	data.write_u8(1).unwrap(); //always a triangle list

	//write the vertex fields
	data.write_u8(0).unwrap();  //Position2D
	data.write_u8(1).unwrap();	//Position3D
	data.write_u8(0).unwrap();	//Color
	data.write_u8( if mesh.format.normal.is_some() { 1 } else { 0 } ).unwrap(); //Normal
	data.write_u8( if mesh.format.tangent.is_some() { 1 } else { 0 } ).unwrap();	//Tangent
	data.write_u8( if mesh.format.tex0.is_some() { 1 } else { 0 } ).unwrap();  //Tex0
	data.write_u8(0).unwrap();	//Tex1

	data.write_f32::<LittleEndian>(mesh.max.x as f32).unwrap();
	data.write_f32::<LittleEndian>(mesh.max.y as f32).unwrap();
	data.write_f32::<LittleEndian>(mesh.max.z as f32).unwrap();

	data.write_f32::<LittleEndian>(mesh.min.x as f32).unwrap();
	data.write_f32::<LittleEndian>(mesh.min.y as f32).unwrap();
	data.write_f32::<LittleEndian>(mesh.min.z as f32).unwrap();

	data.write_u32::<LittleEndian>(mesh.vertices.len() as u32).unwrap();
	data.write_u32::<LittleEndian>(mesh.indices.len() as u32).unwrap();

	for v in mesh.vertices {
		v.write_to(&mut data);
	}

	for idx in mesh.indices {
		match index_size {
			1 => data.write_u8(idx as u8).unwrap(),
			2 => data.write_u16::<LittleEndian>(idx as u16).unwrap(),
			4 => data.write_u32::<LittleEndian>(idx as u32).unwrap(),
			_ => panic!("Invalid index size"),
		}
	}

	data
}

fn convert_obj_set(set: ObjSet, generate_tangents: bool) -> Vec<Vec<u8>> {
	let mut data: Vec<Vec<u8>> = vec![];

	for obj in set.objects {
		data.push(convert_obj(obj, generate_tangents));
	}

	data
}

fn main() {
	let matches = App::new("Obj to mesh converter")
		.version("0.1")
		.about("Still pretty incomplete")
		.arg(Arg::with_name("input")
			.help("The obj file to convert")
			.value_name("OBJ_FILE")
			.takes_value(true)
			.required(true))
		.arg(Arg::with_name("output")
			.long("output")
			.short("o")
			.takes_value(true)
			.value_name("MESH_FILE")
			.help("Sets the output file. Defaults to OBJ_FILE.mesh"))
		.arg(Arg::with_name("gen_tangents")
			.long("gen_tangents")
			.short("t")
			.help("Generates the tangents using UVs"))
		.get_matches();

	let input = Path::new(matches.value_of("input").unwrap());
	
	let output = if let Some(path) = matches.value_of("output") {
		Path::new(path).to_owned()
	}
	else {
		input.with_extension("mesh")
	};

	println!("Converting {} into {}..", 
		input.file_name().unwrap().to_str().unwrap(),
		output.file_name().unwrap().to_str().unwrap()
	);

	let mut file = File::open(input).unwrap();

	let mut content = String::new();
	file.read_to_string(&mut content).unwrap();

	//patch files that don't contain an object name
	if !content.contains("\no") {
		content = "o unnamed_object \n".to_owned() + &content;
	}

	let generate_tangents = matches.occurrences_of("gen_tangents") > 0;

	let data = match wavefront_obj::obj::parse(content) {
	    Ok(obj) => convert_obj_set(obj, generate_tangents),
	    Err(err) => panic!("{:?}", err),
	};

	let mut file = File::create(output).unwrap();

	file.write_all(&data[0]).unwrap();

	println!("Done!");
}
