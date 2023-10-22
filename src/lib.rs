#![doc(html_root_url = "https://docs.rs/wgpu-app/0.17.1")]
//! Rust wgpu partial fork for management Vertex Texture CameraAngle
//!

pub mod vt;
pub mod app;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_vt() {
    assert_eq!(vt::vtx([1, 1, 1], [1, 1]),
      vt::Vertex{pos: [1.0f32; 4], tex_coord: [1.0f32; 2]});
  }

  #[test]
  fn test_app() {
    assert_eq!(app::cast_slice(&[0i8; 4]), &[0u8; 4]);
    assert_eq!(app::cast_slice(&[0u16; 4]), &[0u8; 8]);
    // assert_eq!(app::cast_slice(&[1u16; 2]), &[0u8, 1u8, 0u8, 1u8]); // B.E.
    // assert_eq!(app::cast_slice(&[1u16; 2]), &[1u8, 0u8, 1u8, 0u8]); // L.E.
  }
}
