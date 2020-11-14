#![feature(drain_filter)]
use macroquad::prelude::*;

#[allow(dead_code)]
mod math;
use std::f32::consts::FRAC_PI_2;

macro_rules! or_err {
    ( $r:expr ) => {
        if let Err(e) = $r {
            error!("{}", e)
        }
    };
    ( $l:literal, $r:expr ) => {
        if let Err(e) = $r {
            error!($l, e)
        }
    };
}

macro_rules! system {
    ( $world:ident, $ent:tt, $( $comp:pat = $ab:ty )+ $( > without $($wo:ty, )+ )? $body:block ) => {
        for ($ent, ($($comp,)+)) in &mut $world
            .query::<($($ab,)+)>()
            $( $(.without::<$wo>())* )?
        {
            $body
        }
    };
}

#[derive(Debug, Copy, Clone)]
struct Velocity(Vec2);

#[derive(Debug, Copy, Clone, PartialEq)]
enum CircleKind {
    Push,
    Hurt,
    Hit,
}
impl CircleKind {
    fn hits(self, o: Self) -> bool {
        use CircleKind::*;
        match (self, o) {
            (Push, Push) => true,
            (Hurt, Hit) => true,
            (Hit, Hurt) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Circle(f32, Vec2, CircleKind);
impl Circle {
    fn push(r: f32, offset: Vec2) -> Self {
        Self(r, offset, CircleKind::Push)
    }

    fn hurt(r: f32, offset: Vec2) -> Self {
        Self(r, offset, CircleKind::Hurt)
    }

    fn hit(r: f32, offset: Vec2) -> Self {
        Self(r, offset, CircleKind::Hit)
    }
}

#[derive(Debug, Clone)]
struct Phys(smallvec::SmallVec<[Circle; 5]>);
impl Phys {
    fn new(circles: &[Circle]) -> Self {
        Phys(smallvec::SmallVec::from_slice(circles))
    }
    fn wings(r: f32, wr: f32, kind: CircleKind) -> Self {
        Phys(
            [
                Circle(r, vec2(0.0, r), kind),
                Circle(wr, vec2(-r, r), kind),
                Circle(wr, vec2(r, r), kind),
            ]
            .iter()
            .copied()
            .collect(),
        )
    }
}

#[derive(Debug, Default)]
struct Contacts(smallvec::SmallVec<[Hit; 5]>);
#[derive(Debug)]
struct Hit {
    normal: Vec2,
    depth: f32,
    hit: hecs::Entity,
}

#[derive(Debug, Clone, Copy)]
struct Fixed;

#[derive(Debug, Clone, Copy, Default)]
struct Rot(f32);
impl Rot {
    fn vec2(self) -> Vec2 {
        math::angle_to_vec(self.0)
    }

    fn set_vec2(&mut self, v: Vec2) {
        self.0 = math::vec_to_angle(v)
    }

    fn apply(self, v: Vec2) -> Vec2 {
        let len = v.length();
        let angle = math::vec_to_angle(v);
        math::angle_to_vec(angle + self.0) * len
    }

    fn unapply(self, v: Vec2) -> Vec2 {
        let len = v.length();
        let angle = math::vec_to_angle(v);
        math::angle_to_vec(angle - self.0) * len
    }
}

#[test]
fn apply_unapply() {
    for i in 0..8 {
        use std::f32::consts::TAU;
        let r = Rot(i as f32 / TAU);
        let v = Vec2::unit_y();
        assert!(r.unapply(r.apply(v)).abs_diff_eq(v, 0.001));
    }
}

struct Physics {
    circles: Vec<(hecs::Entity, Vec2, Phys)>,
    collisions: Vec<(hecs::Entity, Hit)>,
    hit_ents: fxhash::FxHashSet<hecs::Entity>,
}
impl Physics {
    fn new() -> Self {
        use {fxhash::FxBuildHasher, std::collections::HashSet};
        Self {
            circles: Vec::with_capacity(1000),
            collisions: Vec::with_capacity(200),
            hit_ents: HashSet::with_capacity_and_hasher(100, FxBuildHasher::default()),
        }
    }

    fn tick(&mut self, ecs: &mut hecs::World) {
        let Self { circles, collisions, hit_ents } = self;

        system!(ecs, _,
            pos           = &mut Vec2
            Velocity(vel) = &mut _
        {
            *pos += *vel;
            *vel *= 0.93;
        });

        circles.extend(
            ecs.query::<(&Vec2, &Phys)>().iter().map(|(e, (&pos, phys))| (e, pos, phys.clone())),
        );
        for (e0, p0, Phys(circles0)) in circles.drain(..) {
            system!(ecs, e1,
                &p1             = &Vec2
                Phys(circles1) = &_
            {
                if e1 == e0 { continue }
                for &Circle(r0, o0, k0) in circles0.iter() {
                    for &Circle(r1, o1, k1) in circles1.iter() {
                        if !k0.hits(k1) { continue }

                        let delta = (p0 + o0) - (p1 + o1);
                        let dist = delta.length();
                        let depth = r0 + r1 - dist;
                        if depth > 0.0 {
                            collisions.push((e0, Hit {
                                hit: e1,
                                depth,
                                normal: delta.normalize(),
                            }));
                            hit_ents.insert(e0);
                        }
                    }
                }
            });
        }

        for &ent in &*hit_ents {
            or_err!(ecs.insert_one(
                ent,
                Contacts(collisions.drain_filter(|(e, _)| *e == ent).map(|(_, h)| h).collect()),
            ));
        }

        system!(ecs, _,
            pos            = &mut Vec2
            Velocity(vel)  = &mut _
            Contacts(hits) = &_
        > without Fixed,
        {
            for &Hit { depth, normal, .. } in hits {
                *pos += normal * depth;
                *vel += normal * depth;
            }
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct EquippedWeapon(hecs::Entity);

#[derive(Debug, Clone, Copy, Default)]
struct WeaponState {
    last_rot: Rot,
}

#[derive(Debug, Clone, Copy)]
struct WeaponInput {
    attacking: bool,
    tick: u32,
    wielder_vel: Vec2,
    wielder_pos: Vec2,
    target: Vec2,
}

#[derive(hecs::Query, Debug)]
struct Weapon<'a> {
    pos: &'a mut Vec2,
    rot: &'a mut Rot,
    wep: &'a mut WeaponState,
    phy: &'a mut Phys,
}
impl Weapon<'_> {
    fn tick(&mut self, input: WeaponInput) {
        for Circle(_, o, _) in self.phy.0.iter_mut() {
            *o = self.wep.last_rot.unapply(*o);
        }

        if input.attacking {
            self.aim(input.wielder_pos, input.target)
        } else {
            self.face(input.wielder_pos, input.wielder_vel, input.tick)
        }

        self.wep.last_rot = *self.rot;
        for Circle(_, o, _) in self.phy.0.iter_mut() {
            *o = self.rot.apply(*o);
        }
    }

    fn face(&mut self, wielder: Vec2, wielder_vel: Vec2, tick: u32) {
        let dir = wielder_vel.x().signum();
        let vl = wielder_vel.length();
        let drag = vl.min(0.07);
        let breathe = (tick as f32 / 35.0).sin() / 30.0;
        let jog = (tick as f32 / 6.85).sin() * vl.min(0.175);
        *self.rot = Rot((FRAC_PI_2 + breathe + jog) * -dir);
        *self.pos = wielder + vec2(
            0.55 * -dir + breathe / 3.2 + jog * 1.5 * -dir + drag * -dir,
            0.35 + (breathe + jog) / 2.8 + drag * 0.5
        );
    }

    fn aim(&mut self, wielder: Vec2, target: Vec2) {
        let center = wielder + vec2(0.0, 0.5);
        let normal = (target - center).normalize();
        *self.pos = center + normal / 2.0;
        *self.rot = Rot(math::vec_to_angle(normal) - FRAC_PI_2);
    }
}

#[derive(hecs::Query, Debug)]
struct Hero<'a> {
    vel: &'a mut Velocity,
    pos: &'a mut Vec2,
    art: &'a Art,
    wep: &'a EquippedWeapon,
}
impl Hero<'_> {
    fn movement(&mut self) {
        #[rustfmt::skip]
        let keymap = [
            (KeyCode::W,  Vec2::unit_y()),
            (KeyCode::S, -Vec2::unit_y()),
            (KeyCode::A, -Vec2::unit_x()),
            (KeyCode::D,  Vec2::unit_x()),
        ];

        let move_vec = keymap
            .iter()
            .filter(|(key, _)| is_key_down(*key))
            .fold(Vec2::zero(), |acc, (_, vec)| acc + *vec)
            .normalize();

        if move_vec.length_squared() > 0.0 {
            self.vel.0 += move_vec * 0.0075;
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Art {
    Hero,
    Target,
    Npc,
    Sword,
}
impl Art {
    fn z_offset(self) -> f32 {
        match self {
            Art::Sword => -1.0,
            _ => 0.0,
        }
    }
}

#[macroquad::main("rpg")]
async fn main() {
    let mut ecs = hecs::World::new();
    let mut drawer = Drawer::new();
    let mut physics = Physics::new();
    let mut fps = Fps::new();

    let wep = ecs.spawn((
        Vec2::zero(),
        Phys::new(&[
            Circle::hurt(0.2, vec2(0.0, 1.35)),
            Circle::hurt(0.185, vec2(0.0, 1.1)),
            Circle::hurt(0.15, vec2(0.0, 0.85)),
            Circle::hurt(0.125, vec2(0.0, 0.65)),
        ]),
        Rot(0.0),
        WeaponState::default(),
        Art::Sword,
    ));
    let hero = ecs.spawn((
        Vec2::unit_y() * -5.0,
        Velocity(Vec2::zero()),
        Phys::wings(0.3, 0.21, CircleKind::Push),
        Art::Hero,
        EquippedWeapon(wep),
    ));

    ecs.spawn((
        Vec2::unit_x() * -4.0,
        Phys::wings(0.3, 0.21, CircleKind::Push),
        Art::Npc,
        Fixed,
    ));

    ecs.spawn((
        Vec2::unit_x() * 4.0,
        Phys::new(&[Circle::push(0.3, vec2(0.0, 0.1)), Circle::hit(0.4, vec2(0.0, 0.4))]),
        Art::Target,
        Fixed,
    ));

    use std::time::{Duration, Instant};
    let step_every = Duration::from_secs_f64(1.0 / 60.0);
    let mut time = step_every;
    let mut step = Instant::now();
    let mut tick: u32 = 0;
    loop {
        time += step.elapsed();
        step = Instant::now();
        while time >= step_every {
            time -= step_every;
            tick = tick.wrapping_add(1);

            physics.tick(&mut ecs);

            let (hero_pos, hero_wep, hero_vel) = match ecs.query_one_mut::<Hero>(hero) {
                Ok(mut hero) => {
                    hero.movement();
                    (*hero.pos, hero.wep.0, hero.vel.0)
                },
                Err(e) => {
                    error!("no hero!? {}", e);
                    continue;
                }
            };
            drawer.pan_cam_to = hero_pos;

            if let Ok(mut wep) = ecs.query_one_mut::<Weapon>(hero_wep) {
                wep.tick(WeaponInput {
                    attacking: is_mouse_button_down(MouseButton::Left),
                    target: drawer.cam.screen_to_world(mouse_position().into()),
                    wielder_pos: hero_pos,
                    wielder_vel: hero_vel,
                    tick,
                });
            }
        }

        drawer.draw(&ecs);
        fps.update();
        fps.render();

        next_frame().await
    }
}

struct Fps {
    fps: [i32; 100],
    pen: usize,
}
impl Fps {
    fn new() -> Self {
        Self { fps: [60; 100], pen: 0 }
    }

    fn update(&mut self) {
        let Self { pen, fps } = self;
        fps[*pen] = get_fps();
        *pen += 1;
        if *pen >= 100 {
            *pen = 0;
        }
    }

    fn average(&self) -> f32 {
        self.fps.iter().map(|i| *i as f32).sum::<f32>() / 100.0
    }

    fn render(&self) {
        draw_text(&self.average().to_string(), 0.0, 0.0, 30.0, BLACK);
    }
}

struct Drawer {
    sprites: Vec<(Vec2, Art, Option<Rot>)>,
    cam: Camera2D,
    pan_cam_to: Vec2,
}
impl Drawer {
    fn new() -> Self {
        Self {
            sprites: Vec::with_capacity(1000),
            cam: Camera2D {
                zoom: vec2(1.0, screen_width() / screen_height()) / 7.8,
                ..Default::default()
            },
            pan_cam_to: Vec2::zero(),
        }
    }

    fn draw(&mut self, ecs: &hecs::World) {
        use std::cmp::Ordering::Less;
        let Self { sprites, cam, .. } = self;
        cam.target = cam.target.lerp(self.pan_cam_to, 0.05);
        set_camera(*cam);

        clear_background(Color([180, 227, 245, 255]));

        let gl = unsafe { get_internal_gl().quad_gl };

        sprites.extend(
            ecs.query::<(&_, &_, Option<&Rot>)>().iter().map(|(_, (&p, &a, r))| (p, a, r.copied())),
        );
        sprites.sort_by(|(pos_a, art_a, ..), (pos_b, art_b, ..)| {
            let a = pos_a.y() + art_a.z_offset();
            let b = pos_b.y() + art_b.z_offset();
            b.partial_cmp(&a).unwrap_or(Less)
        });
        for (pos, art, rot) in sprites.drain(..) {
            const GOLDEN_RATIO: f32 = 1.618034;

            let (color, w, h) = match art {
                Art::Hero => (BLUE, 1.0, 1.0),
                Art::Target => (RED, 0.8, 0.8),
                Art::Npc => (GREEN, 1.0, GOLDEN_RATIO),
                Art::Sword => {
                    gl.push_model_matrix(glam::Mat4::from_translation(glam::vec3(pos.x(), pos.y(), 0.0)));
                    if let Some(Rot(r)) = rot {
                        gl.push_model_matrix(glam::Mat4::from_rotation_z(r));
                    }

                    draw_triangle(
                        vec2( 0.075, 0.0),
                        vec2(-0.075, 0.0),
                        vec2(0.00, GOLDEN_RATIO),
                        BROWN
                    );
                    for &dir in &[-1.0, 1.0] {
                        draw_triangle(
                            vec2(0.00, GOLDEN_RATIO),
                            vec2(dir * 0.1, 0.35),
                            vec2(0.20 * dir, 1.35),
                            GRAY
                        )
                    }
                    draw_triangle(
                        vec2(-0.1, 0.35),
                        vec2( 0.1, 0.35),
                        vec2(0.00, GOLDEN_RATIO),
                        GRAY,
                    );

                    let (x, y) = vec2(-0.225, 0.400).into();
                    let (w, z) = vec2( 0.225, 0.400).into();
                    draw_line(x, y, w, z, 0.135, DARKGRAY);
                    if rot.is_some() {
                        gl.pop_model_matrix();
                    }
                    gl.pop_model_matrix();
                    continue
                },
            };
            let (x, y) = pos.into();
            draw_rectangle(x - w / 2.0, y, w, h, color);
        }

        #[cfg(feature = "show-collide")]
        system!(ecs, _,
            Phys(circles) = &_
            &pos          = &Vec2
        {
            use CircleKind::*;

            fn kind_color(kind: CircleKind) -> Color {
                match kind {
                    Push => PURPLE,
                    Hit => PINK,
                    Hurt => DARKBLUE,
                }
            }

            for &Circle(r, o, k) in circles.iter() {
                let (x, y) = (pos + o).into();
                let mut color = kind_color(k);
                color.0[3] = 100;
                draw_circle(x, y, r, color);
            }

            fn centroid(circles: &[Circle], kind: CircleKind) -> Vec2 {
                let (count, sum) = circles
                    .iter()
                    .filter(|&&Circle(_, _, k)| k == kind)
                    .fold((0, Vec2::zero()), |(i, acc), &Circle(_, p, ..)| (i + 1, acc + p));
                sum / count as f32
            }

            for &kind in &[Push, Hit, Hurt] {
                let (x, y) = (pos + centroid(&circles, kind)).into();
                draw_circle(x, y, 0.1, kind_color(kind));
            }
        });

        set_default_camera();
    }
}
