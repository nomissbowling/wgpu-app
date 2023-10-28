//! vt.rs
//!
//! Vertex index texture FaceInf VIP YRP CameraAngle WG
//!

use std::f32::consts;
use std::error::Error;
use image;
use bytemuck;
use glam;
use wgpu;

/// Vertex must be 4 * (4 4 4 2 1)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
  /// pos (x, y, z, 1.0)
  pub pos: [f32; 4],
  /// norm (x, y, z, 1.0)
  pub norm: [f32; 4],
  /// col (r, g, b, x)
  pub col: [u32; 4],
  /// tex (u, v)
  pub tex_coord: [f32; 2],
  /// ix
  pub ix: u32
}

/// construct Vertex from i8 i8
pub fn vtx(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
  Vertex{
    pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
    norm: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
    col: (0..4).map(|i| if i < 3 { (pos[i] + 2) as u32 * 255 / 4 } else { 255 }
      ).collect::<Vec<_>>().try_into().unwrap(),
    tex_coord: [tc[0] as f32, tc[1] as f32],
    ix: 0
  }
}

/// construct Vertex from i8 f32
pub fn vtf(pos: [i8; 3], tc: [f32; 2]) -> Vertex {
  Vertex{
    pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
    norm: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
    col: (0..4).map(|i| if i < 3 { (pos[i] + 2) as u32 * 255 / 4 } else { 255 }
      ).collect::<Vec<_>>().try_into().unwrap(),
    tex_coord: tc,
    ix: 0
  }
}

/// FaceInf
#[derive(Debug)]
pub struct FaceInf {
  /// number of faces
  pub nfaces: u64,
  /// index list [u64; n + 1]
  pub il: Vec<u64>,
  /// texture function (i, param, max_i) 0 &le; i &lt; max_i
  pub tf: fn ((usize, usize, usize)) -> usize
}

/// Vertex Index Param
#[derive(Debug)]
pub struct VIP {
  /// vertex series
  pub vs: wgpu::Buffer,
  /// index series
  pub is: wgpu::Buffer,
  /// param FaceInf
  pub p: FaceInf
}

/// set location and scale to VIP
pub fn locscale(o: &[f32; 3], scale: f32, vi: (Vec<Vertex>, Vec<u16>, FaceInf))
  -> (Vec<Vertex>, Vec<u16>, FaceInf) {
  (
    vi.0.into_iter().enumerate().map(
      |(j, Vertex{pos: p, col: c, norm: n, tex_coord: tc, ix: _})| {
      Vertex{
        pos: p.iter().enumerate().map(|(i, &v)| {
          if i < 3 { o[i] + v * scale } else { v } // expect else v always 1.0
        }).collect::<Vec<_>>().try_into().unwrap(),
        col: c,
        norm: n,
        tex_coord: tc,
        ix: j as u32
      }
    }).collect(),
    vi.1,
    vi.2
  )
}

/// construct VIP (cube 6 textures)
pub fn create_vertices_cube_6_textures(
  tf: fn ((usize, usize, usize)) -> usize)
  -> (Vec<Vertex>, Vec<u16>, FaceInf) {
  let vertex_data = [ // FrontFace::Ccw culling Face::Back
    // +X (1, 0, 0) right
    vtx([1, -1, 1], [0, 1]),
    vtx([1, -1, -1], [1, 1]),
    vtx([1, 1, -1], [1, 0]),
    vtx([1, 1, 1], [0, 0]),
    // -X (-1, 0, 0) left
    vtx([-1, -1, 1], [0, 1]),
    vtx([-1, 1, 1], [1, 1]),
    vtx([-1, 1, -1], [1, 0]),
    vtx([-1, -1, -1], [0, 0]),
    // +Y (0, 1, 0) back
    vtx([1, 1, -1], [0, 1]),
    vtx([-1, 1, -1], [1, 1]),
    vtx([-1, 1, 1], [1, 0]),
    vtx([1, 1, 1], [0, 0]),
    // -Y (0, -1, 0) front
    vtx([1, -1, -1], [0, 1]),
    vtx([1, -1, 1], [1, 1]),
    vtx([-1, -1, 1], [1, 0]),
    vtx([-1, -1, -1], [0, 0]),
    // +Z (0, 0, 1) top
    vtx([-1, 1, 1], [0, 1]),
    vtx([-1, -1, 1], [1, 1]),
    vtx([1, -1, 1], [1, 0]),
    vtx([1, 1, 1], [0, 0]),

// extra for test
#[cfg(interrupt_vertex)] vtx([0, 2, 0], [0, 1]),
#[cfg(interrupt_vertex)] vtx([2, 2, 0], [1, 1]),
#[cfg(interrupt_vertex)] vtx([2, 0, 0], [1, 0]),
#[cfg(interrupt_vertex)] vtx([0, 0, 0], [0, 0]),

    // -Z (0, 0, -1) bottom
    vtx([-1, 1, -1], [0, 1]),
    vtx([1, 1, -1], [1, 1]),
    vtx([1, -1, -1], [1, 0]),
    vtx([-1, -1, -1], [0, 0])
  ];

  let index_data: &[u16] = &[
    0, 1, 3, 2, 3, 1, // +X right
    4, 5, 7, 6, 7, 5, // -X left
    8, 9, 11, 10, 11, 9, // +Y back
    12, 13, 15, 14, 15, 13, // -Y front
    16, 17, 19, 18, 19, 17, // +Z top

// extra for test
#[cfg(interrupt_vertex)] 24,
#[cfg(interrupt_vertex)] 25,
#[cfg(interrupt_vertex)] 27,
#[cfg(interrupt_vertex)] 26,
#[cfg(interrupt_vertex)] 27,
#[cfg(interrupt_vertex)] 25,

    20, 21, 23, 22, 23, 21 // -Z bottom
  ];

  let nfaces = index_data.len() as u64 / 6;
  (vertex_data.to_vec(), index_data.to_vec(), FaceInf{
    nfaces,
    il: (0..=nfaces).map(|i| i * 6).collect(),
    tf: tf}) // |(i, _, _)| {i} (default)
}

/// construct VIP (cube expansion plan)
pub fn create_vertices_cube_expansion_plan(
  tf: fn ((usize, usize, usize)) -> usize)
  -> (Vec<Vertex>, Vec<u16>, FaceInf) {
  let vertex_data = [ // FrontFace::Ccw culling Face::Back
    // +X (1, 0, 0) right
    vtf([1, -1, 1], [0.25, 0.0]),
    vtf([1, -1, -1], [0.25, 0.25]),
    vtf([1, 1, -1], [0.5, 0.25]),
    vtf([1, 1, 1], [0.5, 0.0]),
    // -X (-1, 0, 0) left
    vtf([-1, -1, 1], [0.25, 0.75]),
    vtf([-1, 1, 1], [0.5, 0.75]),
    vtf([-1, 1, -1], [0.5, 0.5]),
    vtf([-1, -1, -1], [0.25, 0.5]),
    // +Y (0, 1, 0) back
    vtf([1, 1, -1], [0.5, 0.25]),
    vtf([-1, 1, -1], [0.75, 0.25]),
    vtf([-1, 1, 1], [0.75, 0.0]),
    vtf([1, 1, 1], [0.5, 0.0]),
    // -Y (0, -1, 0) front
    vtf([1, -1, -1], [0.0, 0.5]),
    vtf([1, -1, 1], [0.0, 0.75]),
    vtf([-1, -1, 1], [0.25, 0.75]),
    vtf([-1, -1, -1], [0.25, 0.5]),
    // +Z (0, 0, 1) top
    vtf([-1, 1, 1], [0.5, 0.75]),
    vtf([-1, -1, 1], [0.25, 0.75]),
    vtf([1, -1, 1], [0.25, 1.0]),
    vtf([1, 1, 1], [0.5, 1.0]),
    // -Z (0, 0, -1) bottom
    vtf([-1, 1, -1], [0.5, 0.5]),
    vtf([1, 1, -1], [0.5, 0.25]),
    vtf([1, -1, -1], [0.25, 0.25]),
    vtf([-1, -1, -1], [0.25, 0.5])
  ];

  let index_data: &[u16] = &[
    0, 1, 3, 2, 3, 1, // +X right
    4, 5, 7, 6, 7, 5, // -X left
    8, 9, 11, 10, 11, 9, // +Y back
    12, 13, 15, 14, 15, 13, // -Y front
    16, 17, 19, 18, 19, 17, // +Z top
    20, 21, 23, 22, 23, 21 // -Z bottom
  ];

  let nfaces = index_data.len() as u64 / 6;
  (vertex_data.to_vec(), index_data.to_vec(), FaceInf{
    nfaces,
    il: (0..=nfaces).map(|i| i * 6).collect(),
    tf: tf}) // |_| {N} (expansion plan texture number)
}

/// u8 Vec from image file
pub fn load_texels(fname: &str)
  -> Result<((u32, u32, u32, u32), Vec<u8>), Box<dyn Error>> {
  let im = image::open(fname)?; // bpr may be 3b align to 4b
  let (h, w) = (im.height(), im.width());
/*
  print!("load [{}] ", fname);
  if let Some(ref img) = im.as_rgb8() {
    println!("may be rgb {} convert", img.as_raw().len()); // as_raw() &Vec<u8>
  } else {
    println!("may be rgba through");
  }
*/
  let buf = im.into_rgba8().to_vec(); // convert to 4b
  let bytes_per_row = buf.len() as u32 / h; // bpr always align to 4b
  let d = bytes_per_row / w; // expect d is always 4
  println!("{} {} {} {} {}", h, w, d, bytes_per_row, buf.len());
  Ok(((h, w, d, bytes_per_row), buf))
}

/// u8 Vec rgba colors
pub fn create_texels_rgba(size: usize, cols4u: &Vec<&[u8; 4]>)
  -> ((u32, u32, u32, u32), Vec<u8>) {
  // 0u8 black 255u8 max rgba (tex{x, y, z, w} in the shader)
  // bytes_per_row=Some(4 * size) TextureFormat=Rgba8Uint when rgba
  let mut hwd = (size as u32, size as u32, 4, 0);
  hwd.3 = hwd.2 * hwd.1;
  (hwd, (0..(hwd.3 * hwd.0) as usize).map(|id| {
    let bpr = 256 * 4;
    let v = id / bpr;
    let w = id % bpr;
    let u = w / 4;
    (if v < 128 {
      if u < 128 { cols4u[0] } else { cols4u[1] }
    } else {
      if u < 128 { cols4u[2] } else { cols4u[3] }
    })[w % 4]
  }).collect())
}

/// u8 Vec mandelbrot
pub fn create_texels_mandelbrot_4c(size: usize, col4f: &[f32; 4])
  -> ((u32, u32, u32, u32), Vec<u8>) {
  // 0u8 white 255u8 black (1.0 - (v * factor)) in the shader) x 4
  // bytes_per_row=Some(4 * size) TextureFormat=Rgba8Uint when rgba
  // bytes_per_row=Some(size) TextureFormat=R8Uint when gray
  let mut hwd = (size as u32, size as u32, 4, 0);
  hwd.3 = hwd.2 * hwd.1;
  (hwd, (0..(hwd.3 * hwd.0) as usize).map(|id| { // rgba
    let c = id / 4;
    // get high five for recognizing this ;)
    let cx = 3.0 * (c % size) as f32 / (size - 1) as f32 - 2.0;
    let cy = 2.0 * (c / size) as f32 / (size - 1) as f32 - 1.0;
    let (mut x, mut y, mut count) = (cx, cy, 0);
    while count < 0xFF && x * x + y * y < 4.0 {
      let old_x = x;
      x = x * x - y * y + cx;
      y = 2.0 * old_x * y + cy;
      count += 1;
    }
    ((1.0 - (count as f32 / 255.0) * col4f[id % 4]) * 255.0) as u8
  }).collect())
}

/// Yaw Roll Pitch
#[derive(Debug)]
pub struct YRP {
  /// yaw Z axis
  pub yaw: f32,
  /// roll Y axis
  pub roll: f32,
  /// pitch X axis
  pub pitch: f32,
  /// tick
  pub tick: u64
}

/// CameraAngle
#[derive(Debug)]
pub struct CameraAngle {
  /// pos
  pub pos: glam::Vec3,
  /// lookat
  pub lookat: glam::Vec3,
  /// top
  pub top: glam::Vec3
}

/// CameraAngle
impl CameraAngle {
  /// construct
  pub fn new(pos: glam::Vec3, lookat: glam::Vec3, top: glam::Vec3) -> Self {
    CameraAngle{pos, lookat, top}
  }

  /// construct
  pub fn from_yrp(yrp: &YRP) -> Self {
    log::warn!("y:{} r:{} p:{}", yrp.yaw, yrp.roll, yrp.pitch);
    let cs = |t: f32| {let r = t * 3.14159 / 180.0; (r.cos(), r.sin())};
    let (yc, ys) = cs(yrp.yaw);
    let (rc, rs) = cs(yrp.roll);
    let (pc, ps) = cs(yrp.pitch);
    let (tc, _ts) = cs(yrp.tick as f32);
    let radius = (tc + 2.0) * 2.0; // 2.0 <-> 6.0
    let (radius_c, radius_s) = (radius * pc, radius * ps);
    let pos = glam::Vec3::new(radius_c * yc, radius_c * ys, radius_s);
    let lookat = glam::Vec3::ZERO;
    let top = glam::Vec3::new(yc * rs, -ys * rs, rc); // or glam::Vec3::Z or -Z
    log::warn!("{:5.2} {:5.2} {:5.2} {:5.2} {:5.2} {:5.2}",
      pos.x, pos.y, pos.z, top.x, top.y, top.z);
    CameraAngle{pos, lookat, top}
  }

  /// projection view matrix
  pub fn generate_mvp(&self, aspect_ratio: f32) -> glam::Mat4 {
    let projection = glam::Mat4::perspective_rh(
      consts::FRAC_PI_4, aspect_ratio, 1.0, 10.0);
    let view = glam::Mat4::look_at_rh(self.pos, self.lookat, self.top); // _lh
    projection * view
  }
}

/// TexSZ must impl AsRef&lt;[u32; 4]&gt;
#[repr(C)]
#[derive(Debug)]
pub struct TexSZ {
  /// w
  pub w: u32,
  /// h
  pub h: u32,
  /// ext x=mode: 0 square, 1 landscape, 2 portrait, 3 y=max: square
  pub ext: [u32; 2]
}

/// TexSZ
impl AsRef<[u32; 4]> for TexSZ {
  /// as_ref &amp;[u32; 4]
  fn as_ref(&self) -> &[u32; 4] {
unsafe {
    std::slice::from_raw_parts(&self.w as *const u32, 4).try_into().unwrap()
}
  }
}

/// TextureBindGroup (wgpu::BindGroup with texture size)
#[derive(Debug)]
pub struct TextureBindGroup {
  /// wgpu::BindGroup
  pub group: wgpu::BindGroup,
  /// sz (always hold copy)
  pub sz: TexSZ,
  /// wgpu::Buffer for sz
  pub buf: wgpu::Buffer
}

/// World of GL
#[derive(Debug)]
pub struct WG {
  /// VIPs
  pub vips: Vec<VIP>,
  /// vector of TextureBindGroup
  pub bind_group: Vec<TextureBindGroup>,
  /// current bind group
  pub bg: usize,
  /// mvp (always hold copy)
  pub mvp: glam::Mat4,
  /// wgpu::Buffer for mvp
  pub uniform_buf: wgpu::Buffer,
  /// pipeline
  pub pipeline: wgpu::RenderPipeline,
  /// pipeline_wire
  pub pipeline_wire: Option<wgpu::RenderPipeline>,
  /// draw wire without(true) or with(false) texture
  pub wire: bool
}

/// draw_vip
macro_rules! draw_vip {
  ($self: ident, $rp: ident, // self, wgpu::RenderPass,
   $vbuf: ident, ($vs: expr, $ve: expr), // wgpu::Buffer, (u64, u64),
   $ibuf: ident, ($is: expr, $ie: expr), // wgpu::Buffer, (u64, u64),
   $tid: expr, $icnt: expr) => { // usize, u32
    $rp.push_debug_group("Prepare data for draw.");
    $rp.set_pipeline(&$self.pipeline);
    $rp.set_bind_group(0, &$self.bind_group[$tid].group, &[]);
    $rp.set_index_buffer($ibuf.slice($is..$ie), wgpu::IndexFormat::Uint16);
    $rp.set_vertex_buffer(0, $vbuf.slice($vs..$ve));
    $rp.pop_debug_group();
    $rp.insert_debug_marker("Draw!");
    if !&$self.wire {
      $rp.draw_indexed(0..$icnt, 0, 0..1);
    }
    if let Some(ref pipe) = &$self.pipeline_wire {
      $rp.set_pipeline(pipe);
      $rp.draw_indexed(0..$icnt, 0, 0..1);
    }
  }
}

/// World of GL
impl WG {
  /// update matrix
  pub fn update_matrix(
    &mut self,
    config: &wgpu::SurfaceConfiguration,
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    yrp: &YRP) {
    self.mvp = CameraAngle::from_yrp(yrp).generate_mvp(
      config.width as f32 / config.height as f32);
    queue.write_buffer(&self.uniform_buf, 0,
      bytemuck::cast_slice(self.mvp.as_ref())); // &[f32; 16]
  }

  /// draw
  pub fn draw(
    &mut self,
    view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue) {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(
      &wgpu::CommandEncoderDescriptor{label: None});
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
      label: None,
      color_attachments: &[Some(wgpu::RenderPassColorAttachment{
        view,
        resolve_target: None,
        ops: wgpu::Operations{
          load: wgpu::LoadOp::Clear(
            wgpu::Color{r: 0.1, g: 0.2, b: 0.3, a: 1.0}),
          store: true
        }
      })],
      depth_stencil_attachment: None
    });
    for VIP{vs: vertex_buf, is: index_buf, p: fi} in &self.vips {
/*
      // draw faces of vip at a time by one texture
      let isz = std::mem::size_of::<u16>() as u64; // index bytes
      let ipv = index_buf.size() / isz; // indices per vip
      let tid = (fi.tf)((0, self.bg, self.bind_group.len()));
      draw_vip!(self, rpass,
        vertex_buf, (0, vertex_buf.size()),
        index_buf, (0, index_buf.size()),
        tid, ipv as u32);
*/
/**/
      // draw separate face for bind to each texture
      for i in 0..fi.nfaces {
        let isz = std::mem::size_of::<u16>() as u64; // index bytes
//        let ipf = (index_buf.size() / isz) / fi.nfaces; // indices per face
        let ip_s = fi.il[i as usize] * isz; // s * ipf * isz (start of face)
        let ip_e = fi.il[i as usize + 1] * isz; // e * ipf * isz (end of face)
        let ipf = (ip_e - ip_s) / isz; // indices of this face

        let vsz = std::mem::size_of::<Vertex>() as u64; // vertex bytes
        let vpf = (vertex_buf.size() / vsz) / fi.nfaces; // vertices per face
        let (vp_s, vp_e) = (0, fi.nfaces * vpf * vsz); // all vertices

        let tid = (fi.tf)((i as usize, self.bg, self.bind_group.len()));
        draw_vip!(self, rpass,
          vertex_buf, (vp_s, vp_e), // == (..)
          index_buf, (ip_s, ip_e),
          tid, ipf as u32);
      }
/**/
    }
    drop(rpass); // to release encoder
    queue.submit(Some(encoder.finish()));
  }
}
