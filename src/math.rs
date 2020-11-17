use macroquad::prelude::Vec2;

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn inv_lerp(a: f32, b: f32, v: f32) -> f32 {
    (v - a) / (b - a)
}

pub fn remap(from0: f32, to0: f32, from1: f32, to1: f32, t: f32) -> f32 {
    lerp(from1, to1, inv_lerp(from0, to0, t))
}

pub fn smoothstep(t: f32) -> f32 {
    if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        3.0 * t.powi(2) - 2.0 * t.powi(3)
    }
}

/// Shamelessly ported to Rust from:
/// https://www.geometrictools.com/Documentation/FastAndAccurateSlerp.pdf
pub fn slerp(q1: Vec2, q0: Vec2, t: f32) -> Vec2 {
    const MU: f32 = 1.85298109240830;
    const U: [f32; 8] = [
        1.0 / (1.0 * 3.0),
        1.0 / (2.0 * 5.0),
        1.0 / (3.0 * 7.0),
        1.0 / (4.0 * 9.0),
        1.0 / (5.0 * 11.0),
        1.0 / (6.0 * 13.0),
        1.0 / (7.0 * 15.0),
        MU / (8.0 * 17.0),
    ];
    const V: [f32; 8] = [
        1.0 / 3.0,
        2.0 / 5.0,
        3.0 / 7.0,
        4.0 / 9.0,
        5.0 / 11.0,
        6.0 / 13.0,
        7.0 / 15.0,
        MU * 8.0 / 17.0,
    ];

    let xm1 = q0.dot(q1) - 1.0;
    let d = 1.0 - t;
    let t_pow2 = t * t;
    let d_pow2 = d * d;

    let mut ts = [0.0; 8];
    let mut ds = [0.0; 8];
    for i in (0..7).rev() {
        ts[i] = (U[i] * t_pow2 - V[i]) * xm1;
        ds[i] = (U[i] * d_pow2 - V[i]) * xm1;
    }

    #[rustfmt::skip]
    let f0 = t
        * (1.0 + ts[0] * (1.0 + ts[1] * (1.0 + ts[2] * (1.0 + ts[3] *
          (1.0 + ts[4] * (1.0 + ts[5] * (1.0 + ts[6] * (1.0 + ts[7]))))))));

    #[rustfmt::skip]
    let f1 = d
        * (1.0 + ds[0] * (1.0 + ds[1] * (1.0 + ds[2] * (1.0 + ds[3] *
          (1.0 + ds[4] * (1.0 + ds[5] * (1.0 + ds[6] * (1.0 + ds[7]))))))));

    q0 * f0 + q1 * f1
}
