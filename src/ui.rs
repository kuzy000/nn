use itertools::Itertools;
use macroquad::{
    prelude::*,
    ui::{hash, root_ui},
};

use std::sync::{Arc, Mutex};

use crate::ui_state;
use ui_state::UiState;

fn draw(state: &mut UiState) {
    clear_background(WHITE);

    let loss = *state.loss.last().unwrap_or(&-1.);

    let ui = &mut root_ui();

    ui.label(None, &format!("FPS: {}", get_fps()));
    ui.checkbox(hash!(), "Pause", &mut state.pause);
    ui.slider(hash!(), "Rate", 0.000001..0.1, &mut state.rate);
    ui.slider(hash!(), "Img", 0.0..1.0, &mut state.slider);
    ui.label(None, &format!("Epoch: {}", state.epoch));
    ui.label(None, &format!("Loss: {}", loss));

    'out: {
        let s = vec2(300., 300.);
        let mut canvas = ui.canvas();
        let p = canvas.request_space(s);

        canvas.rect(Rect::new(p.x, p.y, s.x, s.y), BLACK, None);
        
        let size = 250;

        let max = *state
            .loss
            .iter()
            .rev()
            .take(size)
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or(&0.);

        if state.loss.len() < 2 {
            break 'out;
        }

        let num = state.loss.len().min(size);
        let beg = state.loss.len().saturating_sub(size);

        for i in 0..num - 1 {
            let xa = (i + 0) as f32 / (num - 1) as f32;
            let xb = (i + 1) as f32 / (num - 1) as f32;
            
            let ia = beg + i + 0;
            let ib = beg + i + 1;

            let ya = state.loss.get(ia).unwrap();
            let yb = state.loss.get(ib).unwrap();

            canvas.line(
                p + vec2(xa, 1. - ya / max) * s,
                p + vec2(xb, 1. - yb / max) * s,
                BLACK,
            );
        }

//        let num = state.loss.len().min(200);
//
//        for i in 0..num - 1 {
//            let xa = (i + 0) as f32 / (num - 1) as f32;
//            let xb = (i + 1) as f32 / (num - 1) as f32;
//            
//            let factor = state.loss.len() as f32 / num as f32;
//            
//            let ia = ((i + 0) as f32 * factor) as usize;
//            let ib = ((i + 1) as f32 * factor) as usize;
//
//            let ya = state.loss.get(ia).unwrap();
//            let yb = state.loss.get(ib).unwrap();
//
//            canvas.line(
//                p + vec2(xa, 1. - ya / max) * s,
//                p + vec2(xb, 1. - yb / max) * s,
//                BLACK,
//            );
//        }
    };
    
    if !state.img.is_empty() {
        let buf: Vec<u8> = state.img.iter().map(|v| {
            let v = ((1. - v) * 255.).clamp(0., 255.) as u8;
            [v, v, v, 255]
        }).flatten().collect();
        
        let mut buf = Vec::<u8>::new();
        buf.resize(state.img.len() / 3 * 4, 0);

        { 
            let mut j = 0;
            for i in (0..state.img.len()).step_by(3) {
                let s = &state.img;
                let (r, g, b) = (s[i + 0], s[i + 1], s[i + 2]);
                
                buf[j + 0] = (r * 255.).clamp(0., 255.) as u8;
                buf[j + 1] = (g * 255.).clamp(0., 255.) as u8;
                buf[j + 2] = (b * 255.).clamp(0., 255.) as u8;
                buf[j + 3] = 255;

                j += 4;
            }
        }

        let texture = Texture2D::from_rgba8(state.size.0 as u16, state.size.1 as u16, &buf);
        texture.set_filter(FilterMode::Nearest);
        
        let dp = DrawTextureParams { 
            dest_size: Some(vec2(256., 256.)),
            ..Default::default()
        };

        draw_texture_ex(&texture, 400., 100., WHITE, dp);
    }
}

async fn async_ui_main(ui_state: Arc<Mutex<UiState>>) -> Result<(), macroquad::Error> {
    loop {
        draw(ui_state.lock().as_mut().unwrap());
        next_frame().await
    }
}

pub fn ui_main(ui_state: Arc<Mutex<UiState>>) {
    macroquad::Window::new("nn", async {
        async_ui_main(ui_state).await.unwrap();
    });
}
