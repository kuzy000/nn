pub mod ui;
pub mod ui_state;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::{AutodiffBackend, Backend};

use burn::tensor::{Data, Shape, Tensor};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use itertools::Itertools;
use ui::ui_main;
use ui_state::UiState;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    ln: Vec<Linear<B>>,
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
        let mut xs = input.clone();
        for l in &self.ln {
            //xs = l.forward(&xs.tanh()?)?;
            xs =  relu(l.forward(xs));
        }
        //xs.clamp(0., 1.)
        //sigmoid(xs)
        xs.clamp(0., 1.)
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

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    main_backend::<burn::backend::Autodiff<burn::backend::NdArray>>(device)
}


fn main_backend<B: AutodiffBackend<FloatElem = f32>>(device: B::Device) -> Result<()> {
    let ui_state = Arc::new(Mutex::new(UiState::new()));
    {
        let s = ui_state.clone();
        thread::spawn(move || ui_main(s));
    }

    let image = image::io::Reader::open("2.png")?.decode()?;
    let image = image.grayscale().to_luma32f();
    let w = image.width();
    let h = image.height();
    
    let image = image.to_vec().iter().map(|x| 1. - x).collect();

    let coords: Vec<f32> = (0..h)
        .cartesian_product(0..w)
        .map(|(y, x)| [(y as f32 / (h - 1) as f32), (x as f32 / (w - 1) as f32)])
        .flatten()
        .collect();

    // let tw = 256;
    // let th = 256;

    // let coords_th: Vec<f32> = (0..th)
    //     .cartesian_product(0..tw)
    //     .map(|(y, x)| [(y as f32 / th as f32) * 2. - 1., (x as f32 / tw as f32) * 2. - 1.])
    //     .flatten()
    //     .collect();
    
    let ti = Tensor::<B, 2>::from_floats(Data::new(coords, Shape::new([(w * h) as usize, 2])), &device);
    // let ti = Tensor::from_vec(coords, ((w * h) as usize, 2), &device)?.to_dtype(candle_core::DType::F16)?;

    let to = Tensor::<B, 2>::from_floats(Data::new(image, Shape::new([(w * h) as usize, 1])), &device);
    // let to = Tensor::from_slice(&image, ((w * h) as usize, 1), &device)?.to_dtype(candle_core::DType::F16)?;

    // let tt = Tensor::<B, 2>::from_floats(Data::new(coords_th, Shape::new([(tw * th) as usize, 2])), &device);
    //let tt = Tensor::from_vec(coords_th, ((tw * th) as usize, 2), &device)?.to_dtype(candle_core::DType::F16)?;

    let mut model = Model::new(&[2, 20, 20, 20, 1], &device)?;

    ui_state.lock().unwrap().rate = 0.005;
    //let mut sgd = SGD::new(varmap.all_vars(), ui_state.lock().unwrap().rate as f64)?;
    
    let sdg_config = SgdConfig::new();
    let mut sgd = sdg_config.init();

    for i in 1.. {
        let f = model.forward(ti.clone());
        let loss = f.clone().sub(to.clone()).powf_scalar(2.).sum().sqrt();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        
        model = sgd.step(ui_state.lock().unwrap().rate as f64, model, grads);
        
        // if i % 5 != 0 {
        //     continue;
        // }
        
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

        let res = f.to_data().value;
        ui_state.lock().as_mut().unwrap().img = res;

        while ui_state.lock().unwrap().pause {
            std::thread::sleep(Duration::from_secs_f32(1. / 60.));
        }
        //std::thread::sleep(Duration::from_secs_f32(0.01));
    }

    Ok(())
}