use std::str::FromStr;

pub struct UiState {
    pub title: String,
    pub pause: bool,
    pub loss: Vec<f32>,
    pub img: Vec<f32>,
    pub rate: f32,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            title: String::from_str("nn").unwrap(),
            pause: false,
            loss: Vec::new(),
            img: Vec::new(),
            rate: 0.05,
        }
    }
}