//! app.rs
//!
//! partial fork (remove wasm32) from
//!
//! https://github.com/gfx-rs/wgpu/blob/v0.17/examples/common/src/framework.rs
//!

use std::future::Future;
use std::time::Instant;
use winit::event::{self, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use wgpu;

// #[allow(dead_code)]
pub fn cast_slice<T>(data: &[T]) -> &[u8] {
  use std::{mem::size_of, slice::from_raw_parts};
unsafe {
  from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>())
}
}

// #[allow(dead_code)]
pub enum ShaderStage {
  Vertex,
  Fragment,
  Compute
}

pub trait App: 'static + Sized {
  fn optional_features() -> wgpu::Features {
    wgpu::Features::empty()
  }
  fn required_features() -> wgpu::Features {
//    wgpu::Features::empty()
    wgpu::Features::DEPTH_CLIP_CONTROL
  }
  fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
    wgpu::DownlevelCapabilities{
      flags: wgpu::DownlevelFlags::empty(),
      shader_model: wgpu::ShaderModel::Sm5,
      ..wgpu::DownlevelCapabilities::default()
    }
  }
  // These downlevel limits will allow the code to run on all possible hardware
  fn required_limits() -> wgpu::Limits {
    wgpu::Limits::downlevel_webgl2_defaults()
  }

  fn init(
    config: &wgpu::SurfaceConfiguration,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue
  ) -> Self;
  fn resize(
    &mut self,
    config: &wgpu::SurfaceConfiguration,
    device: &wgpu::Device,
    queue: &wgpu::Queue
  );
  fn update(&mut self,
    event: WindowEvent,
    config: &wgpu::SurfaceConfiguration,
    device: &wgpu::Device,
    queue: &wgpu::Queue
  );
  fn render(
    &mut self,
    view: &wgpu::TextureView,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spawner: &Spawner
  );
}

struct Setup {
  window: winit::window::Window,
  event_loop: EventLoop<()>,
  instance: wgpu::Instance,
  size: winit::dpi::PhysicalSize<u32>,
  surface: wgpu::Surface,
  adapter: wgpu::Adapter,
  device: wgpu::Device,
  queue: wgpu::Queue
}

async fn setup<E: App>(title: &str) -> Setup {
  env_logger::init(); // with env set RUST_LOG=error warn info debug trace

  let event_loop = EventLoop::new();
  let mut builder = winit::window::WindowBuilder::new();
  builder = builder.with_title(title);
  #[cfg(windows_OFF)] // TODO
  {
    use winit::platform::windows::WindowBuilderExtWindows;
    builder = builder.with_no_redirection_bitmap(true);
  }
  let window = builder.build(&event_loop).unwrap();

  log::info!("Initializing the surface...");

  let backends = wgpu::util::backend_bits_from_env()
    .unwrap_or_else(wgpu::Backends::all);
  let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env()
    .unwrap_or_default();

  let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
    backends,
    dx12_shader_compiler
  });
  let (size, surface) = unsafe {
    let size = window.inner_size();
    let surface = instance.create_surface(&window).unwrap();
    (size, surface)
  };
  let adapter = wgpu::util::initialize_adapter_from_env_or_default(
    &instance, Some(&surface))
    .await
    .expect("No suitable GPU adapters found on the system!");

  let adapter_info = adapter.get_info();
  println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

  let optional_features = E::optional_features();
  let required_features = E::required_features();
  let adapter_features = adapter.features();
  assert!(
    adapter_features.contains(required_features),
    "Adapter does not support required features for this App: {:?}",
    required_features - adapter_features
  );

  let required_downlevel_capabilities = E::required_downlevel_capabilities();
  let downlevel_capabilities = adapter.get_downlevel_capabilities();
  assert!(
    downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
    "Adapter does not support the minimum shader model required to run this App: {:?}",
    required_downlevel_capabilities.shader_model
  );
  assert!(
    downlevel_capabilities
      .flags
      .contains(required_downlevel_capabilities.flags),
    "Adapter does not support the downlevel capabilities required to run this App: {:?}",
    required_downlevel_capabilities.flags - downlevel_capabilities.flags
  );

  // Make sure we use the texture resolution limits from the adapter,
  // so we can support images the size of the surface.
  let needed_limits = E::required_limits().using_resolution(adapter.limits());

  let trace_dir = std::env::var("WGPU_TRACE");
  let (device, queue) = adapter
    .request_device(
      &wgpu::DeviceDescriptor{
        label: None,
        features: (optional_features & adapter_features) | required_features,
        limits: needed_limits
      },
      trace_dir.ok().as_ref().map(std::path::Path::new)
    )
    .await
    .expect("Unable to find a suitable GPU adapter!");

  Setup{
    window,
    event_loop,
    instance,
    size,
    surface,
    adapter,
    device,
    queue
  }
}

fn start<E: App>(
  Setup{
    window,
    event_loop,
    instance,
    size,
    surface,
    adapter,
    device,
    queue
  }: Setup
) {
  let spawner = Spawner::new();
  let mut config = surface
    .get_default_config(&adapter, size.width, size.height)
    .expect("Surface isn't supported by the adapter.");
  let surface_view_format = config.format.add_srgb_suffix();
  config.view_formats.push(surface_view_format);
  surface.configure(&device, &config);

  log::info!("Initializing the App...");
  let mut app = E::init(&config, &adapter, &device, &queue);

  let mut last_frame_inst = Instant::now();
  let (mut frame_count, mut accum_time) = (0, 0.0);

  log::info!("Entering render loop...");
  event_loop.run(move |event, _, control_flow| {
    let _ = (&instance, &adapter); // force ownership by the closure
    *control_flow = if cfg!(feature = "metal-auto-capture") {
      ControlFlow::Exit
    } else {
      ControlFlow::Poll
    };
    match event {
    event::Event::RedrawEventsCleared => {
      spawner.run_until_stalled();
      window.request_redraw();
    },
    event::Event::WindowEvent {
      event: WindowEvent::Resized(size) | WindowEvent::ScaleFactorChanged{
        new_inner_size: &mut size,
        ..
      },
      ..
    } => {
      // Once winit is fixed, the detection conditions here can be removed.
      // https://github.com/rust-windowing/winit/issues/2876
      let max_dimension = adapter.limits().max_texture_dimension_2d;
      if size.width > max_dimension || size.height > max_dimension {
        log::warn!(
          "The resizing size {:?} exceeds the limit of {}.",
          size,
          max_dimension
        );
      } else {
        log::info!("Resizing to {:?}", size);
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        app.resize(&config, &device, &queue);
        surface.configure(&device, &config);
      }
    },
    event::Event::WindowEvent{ event, .. } => {
      match event {
      WindowEvent::CloseRequested | WindowEvent::KeyboardInput{
        input: event::KeyboardInput{
          virtual_keycode: Some(event::VirtualKeyCode::Escape),
          state: event::ElementState::Pressed,
          ..
        },
        ..
      } => *control_flow = ControlFlow::Exit,
      WindowEvent::KeyboardInput{
        input: event::KeyboardInput{
          virtual_keycode: Some(event::VirtualKeyCode::R),
          state: event::ElementState::Pressed,
          ..
        },
        ..
      } => println!("{:#?}", instance.generate_report()),
      _ => app.update(event, &config, &device, &queue)
      }
    },
    event::Event::RedrawRequested(_) => {
      accum_time += last_frame_inst.elapsed().as_secs_f32();
      last_frame_inst = Instant::now();
      frame_count += 1;
      if frame_count == 100 * 60 { // 100
        println!(
          "Avg frame time {}ms",
          accum_time * 1000.0 / frame_count as f32
        );
        accum_time = 0.0;
        frame_count = 0;
      }

      let frame = match surface.get_current_texture() {
      Ok(frame) => frame,
      Err(_) => {
        surface.configure(&device, &config);
        surface.get_current_texture()
          .expect("Failed to acquire next surface texture!")
      }
      };
      let view = frame.texture.create_view(&wgpu::TextureViewDescriptor{
        format: Some(surface_view_format),
        ..wgpu::TextureViewDescriptor::default()
      });

      app.render(&view, &device, &queue, &spawner);

      frame.present();
    },
    _ => {}
    }
  });
}

pub struct Spawner<'a> {
  executor: async_executor::LocalExecutor<'a>
}

impl<'a> Spawner<'a> {
  fn new() -> Self {
    Self{ executor: async_executor::LocalExecutor::new() }
  }

  // #[allow(dead_code)]
  pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
    self.executor.spawn(future).detach();
  }

  fn run_until_stalled(&self) {
    while self.executor.try_tick() {}
  }
}

pub fn run<E: App>(title: &str) {
  let setup = pollster::block_on(setup::<E>(title));
  start::<E>(setup);
}
