use std::str::FromStr;

pub struct UiState {
    pub title: String,
    pub pause: bool,
    pub loss: Vec<f32>,
    pub img: Vec<f32>,
    pub rate: f32,
    pub slider: f32,
    pub epoch: i32,
    pub size: (u32, u32)
}

impl UiState {
    pub fn new() -> Self {
        Self {
            title: String::from_str("nn").unwrap(),
            pause: false,
            loss: Vec::new(),
            img: Vec::new(),
            rate: 0.05,
            slider: 0.,
            epoch: -1,
            size: (32, 32),
        }
    }
}