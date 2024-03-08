pub mod ui;
pub mod ui_state;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::{AutodiffBackend, Backend};

use burn::tensor::{Data, ElementConversion, Shape, Tensor};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use itertools::{concat, Itertools};
use ui::ui_main;
use ui_state::UiState;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    ln: Vec<Linear<B>>,
}

pub fn leaky_relu<const D: usize, B: Backend, E: ElementConversion>(
    tensor: Tensor<B, D>,
    slope: E,
) -> Tensor<B, D> {
    let leak = tensor.clone().clamp_min(0.).mul_scalar(slope);
    tensor.clamp_min(0.) + leak
}

impl<B: Backend> Model<B> {
    fn new(arch: &[usize], device: &B::Device) -> Result<Self> {
        Ok(Model {
            ln: arch
                .iter()
                .tuple_windows()
                .enumerate()
                .map(|(i, (a, b))| LinearConfig::new(*a, *b).init(device))
                .collect(),
        })
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let slope = 0.1;

        let mut xs = input.clone();
        for i in (0..&self.ln.len() - 1) {
            let l = &self.ln[i];
            xs = leaky_relu(l.forward(xs), slope);
        }
        let l = &self.ln.last().unwrap();
        xs = sigmoid(l.forward(xs));

        xs
    }
}

struct StrideIter<I> {
    iter: I,
    stride: usize,
    width: usize,
    pos: usize,
}

impl<I> StrideIter<I> {
    fn new(iter: I, stride: usize, width: usize) -> Self {
        assert!(stride <= width);
        StrideIter {
            iter: iter,
            stride: stride,
            width: width,
            pos: 0,
        }
    }
}

impl<I: Iterator> Iterator for StrideIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.stride {
            self.pos += 1;
            self.iter.next()
        } else {
            self.pos = 1;
            self.iter.nth(self.width - self.stride)
        }
    }
}

fn main() {
    match try_main() {
        Err(e) => print!("{e}"),
        _ => (),
    }
}

fn try_main() -> Result<()> {
    // let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    // main_backend::<burn::backend::Autodiff<burn::backend::LibTorch>>(device)

    // let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    // main_backend::<burn::backend::Autodiff<burn::backend::NdArray>>(device)

    let device = burn::backend::wgpu::WgpuDevice::default();
    main_backend::<burn::backend::Autodiff<burn::backend::Wgpu>>(device)
}

fn img_grayscale(s: &str) -> Result<(Vec<f32>, u32, u32)> {
    let image = image::io::Reader::open(s)?.decode()?;
    let image = image.grayscale().to_luma32f();
    let w = image.width();
    let h = image.height();
    let image = image.to_vec().iter().map(|x| 1. - x).collect();

    Ok((image, w, h))
}

fn img_yuv(s: &str) -> Result<(Vec<f32>, u32, u32)> {
    let image = image::io::Reader::open(s)?.decode()?;
    let image = image.to_rgb32f();
    let w = image.width();
    let h = image.height();
    let mut image = image.to_vec();
    transform_color(&mut image, rgb_to_yuv);

    Ok((image, w, h))
}

fn transform_color<F>(image: &mut Vec<f32>, f: F)
where
    F: Fn(f32, f32, f32) -> [f32; 3],
{
    for i in (0..image.len()).step_by(3) {
        let (r, g, b) = (image[i + 0], image[i + 1], image[i + 2]);
        let [y, u, v] = f(r, g, b);
        image[i + 0] = y;
        image[i + 1] = u;
        image[i + 2] = v;
    }
}

fn map_range(a0: f32, b0: f32, a1: f32, b1: f32, v: f32) -> f32 {
    a1 + (v - a0) / (b0 - a0) * (b1 - a1)
}

fn make_coords(w: u32, h: u32, third: f32, offset: f32) -> Vec<f32> {
    (0..h)
        .cartesian_product(0..w)
        .map(|(y, x)| {
            [
                map_range(0., (h - 1) as f32, -1. - offset, 1. + offset, y as f32),
                map_range(0., (w - 1) as f32, -1. - offset, 1. + offset, x as f32),
                third,
            ]
        })
        .flatten()
        .collect()
}

fn rgb_to_yuv(r: f32, g: f32, b: f32) -> [f32; 3] {
    [
        r * 0.2126 + g * 0.7152 + b * 0.0722,
        r * -0.09991 + g * -0.33609 + b * 0.436,
        r * 0.615 + g * -0.55861 + b * -0.05639,
    ]
}

fn yuv_to_rgb(y: f32, u: f32, v: f32) -> [f32; 3] {
    [
        y * 1.0 + u * 0.0 + v * 1.28033,
        y * 1.0 + u * -0.21482 + v * -0.38059,
        y * 1.0 + u * 2.12798 + v * 0.0,
    ]
}

fn main_backend<B: AutodiffBackend<FloatElem = f32>>(device: B::Device) -> Result<()> {
    let ui_state = Arc::new(Mutex::new(UiState::new()));
    {
        let s = ui_state.clone();
        thread::spawn(move || ui_main(s));
    }

    let (image0, w, h) = img_yuv("furby.png")?;
    let (image1, _, _) = img_yuv("furby2.png")?;

    let coords0 = make_coords(w, h, 0., 0.);
    let coords1 = make_coords(w, h, 1., 0.);

    let coords = {
        let mut t = coords0.clone();
        t.append(&mut coords1.clone());
        t
    };
    let image = {
        let mut t = image0.clone();
        t.append(&mut image1.clone());
        t
    };

    //println!("{:?}", coords);

    // for i in (0..coords.len()).step_by(3) {
    //     let a = coords[i + 0];
    //     let b = coords[i + 1];
    //     let c = coords[i + 2];

    //     println!("{a:.3} {b:.3} {c:.3}");
    // }

    // let tw = 256;
    // let th = 256;

    // let coords_th: Vec<f32> = (0..th)
    //     .cartesian_product(0..tw)
    //     .map(|(y, x)| [(y as f32 / th as f32) * 2. - 1., (x as f32 / tw as f32) * 2. - 1.])
    //     .flatten()
    //     .collect();

    assert_eq!(coords.len(), (w * h * 2 * 3) as usize);
    let ti = Tensor::<B, 2>::from_floats(
        Data::new(coords, Shape::new([(w * h * 2) as usize, 3])),
        &device,
    );
    // let ti = Tensor::from_vec(coords, ((w * h) as usize, 2), &device)?.to_dtype(candle_core::DType::F16)?;

    assert_eq!(image.len(), (w * h * 2 * 3) as usize);
    let to = Tensor::<B, 2>::from_floats(
        Data::new(image, Shape::new([(w * h * 2) as usize, 3])),
        &device,
    );
    // let to = Tensor::from_slice(&image, ((w * h) as usize, 1), &device)?.to_dtype(candle_core::DType::F16)?;

    // let tt = Tensor::<B, 2>::from_floats(Data::new(coords_th, Shape::new([(tw * th) as usize, 2])), &device);
    //let tt = Tensor::from_vec(coords_th, ((tw * th) as usize, 2), &device)?.to_dtype(candle_core::DType::F16)?;

    // let mut model = Model::new(&[2, 20, 20, 20, 1], &device)?;
    let mut model = Model::new(&[3, 200, 200, 200, 200, 200, 3], &device)?;

    ui_state.lock().unwrap().rate = 0.001;
    //let mut sgd = SGD::new(varmap.all_vars(), ui_state.lock().unwrap().rate as f64)?;

    //let sdg_config = SgdConfig::new();
    //let mut opt = sdg_config.init();
    
    let adam_config = AdamConfig::new();
    let mut opt = adam_config.init();

    for i in 1.. {
        if !ui_state.lock().unwrap().pause {
            let f = model.forward(ti.clone());
            let loss = f.clone().sub(to.clone()).powf_scalar(2.).sum().sqrt();

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            model = opt.step(ui_state.lock().unwrap().rate as f64, model, grads);


            // if i % 1000 == 0 {
            //     let tf = model.forward(&tt)?;
            //     let res: Vec<f32> = tf.squeeze(1)?.to_dtype(candle_core::DType::F32)?.to_vec1()?;
            //     let res = res.iter().map(|x| (x * 255.) as u8).collect::<Vec<u8>>();

            //     let image: GrayImage = ImageBuffer::from_raw(tw, th, res).unwrap();
            //     image.save("out.png")?;
            // }

            let loss = loss.into_scalar();

            // sgd.set_learning_rate((ui_state.lock().unwrap().rate / loss) as f64);

            // assert!(!loss.is_nan());

            ui_state.lock().unwrap().loss.push(loss);
            ui_state.lock().as_mut().unwrap().epoch = i;
        }

        if i % 10 != 0 || ui_state.lock().unwrap().pause {
            let w = w * 4;
            let h = h * 4;
            ui_state.lock().as_mut().unwrap().size = (w, h);

            let slider = ui_state.lock().unwrap().slider;
            let c = make_coords(w, h, slider, 0.1);
            let ti = Tensor::<B, 2>::from_floats(
                Data::new(c, Shape::new([(w * h) as usize, 3])),
                &device,
            );
            let f = model.forward(ti.clone());

            let mut res = f.to_data().value;
            transform_color(&mut res, yuv_to_rgb);
            ui_state.lock().as_mut().unwrap().img = res;
        }

        if ui_state.lock().unwrap().pause {
            std::thread::sleep(Duration::from_millis(2));
        }
        //std::thread::sleep(Duration::from_secs_f32(0.01));
    }

    Ok(())
}
