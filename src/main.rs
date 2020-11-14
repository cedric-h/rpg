#![feature(drain_filter)]
use macroquad::prelude::*;

#[allow(dead_code)]
mod math;
use std::f32::consts::FRAC_PI_2;

macro_rules! or_err {
    ( $r:expr ) => {
        if let Err(e) = $r {
            error!("[{} {}:{}]: {}", file!(), line!(), column!(), e)
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

#[derive(Debug, Default, Clone)]
struct Contacts(smallvec::SmallVec<[Hit; 5]>);
#[derive(Debug, Clone, Copy)]
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
    stale_hit_ents: Vec<hecs::Entity>,
}
impl Physics {
    fn new() -> Self {
        use {fxhash::FxBuildHasher, std::collections::HashSet};
        Self {
            circles: Vec::with_capacity(1000),
            collisions: Vec::with_capacity(200),
            hit_ents: HashSet::with_capacity_and_hasher(100, FxBuildHasher::default()),
            stale_hit_ents: Vec::with_capacity(100),
        }
    }

    fn tick(&mut self, ecs: &mut hecs::World) {
        let Self { circles, collisions, hit_ents, stale_hit_ents } = self;

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
            if let Err(hecs::NoSuchEntity) = ecs.insert_one(
                ent,
                Contacts(collisions.drain_filter(|(e, _)| *e == ent).map(|(_, h)| h).collect()),
            ) {
                stale_hit_ents.push(ent);
            }
        }

        for ent in stale_hit_ents.drain(..) {
            hit_ents.remove(&ent);
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

#[derive(Debug, Clone, Default)]
struct WeaponState {
    last_rot: Rot,
    attack: Attack,
    ents_hit: smallvec::SmallVec<[hecs::Entity; 5]>,
}

#[derive(Debug, Clone, Copy)]
enum Attack {
    Ready,
    Swing(SwingState),
    Cooldown { end_tick: u32 },
}
impl Default for Attack {
    fn default() -> Self {
        Attack::Ready
    }
}

#[derive(Debug, Clone, Copy)]
struct SwingState {
    doing_damage: bool,
    start_tick: u32,
    end_tick: u32,
    target: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct WeaponInput {
    start_attacking: bool,
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
    contacts: &'a Contacts,
}
impl Weapon<'_> {
    fn tick(&mut self, input: WeaponInput) {
        for Circle(_, o, _) in self.phy.0.iter_mut() {
            *o = self.wep.last_rot.unapply(*o);
        }

        if self.aim(input) {
            let mut atk = std::mem::take(&mut self.wep.attack);
            if let Attack::Swing(state) = &mut atk {
                state.doing_damage = self.swing(input, *state);
            }
            self.wep.attack = atk;
        } else {
            let (rot, pos) = self.rest(input);
            *self.rot = rot;
            *self.pos = pos;
        }

        self.wep.last_rot = *self.rot;
        for Circle(_, o, _) in self.phy.0.iter_mut() {
            *o = self.rot.apply(*o);
        }
    }

    fn damage_ents(&mut self) -> smallvec::SmallVec<[hecs::Entity; 5]> {
        match self.wep.attack {
            Attack::Swing(s) if s.doing_damage => {
                self
                    .contacts
                    .0
                    .iter()
                    .map(|h| h.hit)
                    .filter(|e| {
                        if self.wep.ents_hit.contains(e) {
                            false
                        } else {
                            self.wep.ents_hit.push(*e);
                            true
                        }
                    })
                    .collect()
            },
            _ => Default::default(),
        }
    }

    fn rest(
        &mut self,
        WeaponInput { wielder_pos, wielder_vel, tick, .. }: WeaponInput,
    ) -> (Rot, Vec2) {
        let dir = wielder_vel.x().signum();
        let vl = wielder_vel.length();
        let drag = vl.min(0.07);
        let breathe = (tick as f32 / 35.0).sin() / 30.0;
        let jog = (tick as f32 / 6.85).sin() * vl.min(0.175);
        (
            Rot((FRAC_PI_2 + breathe + jog) * -dir),
            wielder_pos
                + vec2(
                    0.55 * -dir + breathe / 3.2 + jog * 1.5 * -dir + drag * -dir,
                    0.35 + (breathe + jog) / 2.8 + drag * 0.5,
                ),
        )
    }

    fn aim(&mut self, input: WeaponInput) -> bool {
        let WeaponInput { start_attacking, target, tick, .. } = input;

        use Attack::*;
        self.wep.attack = match self.wep.attack {
            Ready if start_attacking => {
                self.wep.ents_hit.clear();
                Swing(SwingState {
                    start_tick: tick,
                    end_tick: tick + 50,
                    target,
                    doing_damage: false,
                })
            },
            Swing(SwingState { end_tick, .. }) if end_tick < tick => {
                self.wep.ents_hit.clear();
                Cooldown { end_tick: tick + 10 }
            }
            Cooldown { end_tick, .. } if end_tick < tick => Ready,
            other => other,
        };

        !matches!(self.wep.attack, Ready | Cooldown { .. })
    }

    fn center(&mut self, wielder_pos: Vec2) -> Vec2 {
        wielder_pos + vec2(0.0, 0.5)
    }

    fn from_center(&mut self, wielder_pos: Vec2, to: Vec2) -> Vec2 {
        (to - self.center(wielder_pos)).normalize()
    }

    fn swing(
        &mut self,
        input: WeaponInput,
        SwingState { start_tick, end_tick, target, .. }: SwingState,
    ) -> bool {
        let WeaponInput { wielder_pos, tick, .. } = input;
        let (start_rot, start_pos) = self.rest(input);

        let center = self.center(wielder_pos);
        let normal = self.from_center(wielder_pos, target);
        let rot = math::vec_to_angle(normal) - FRAC_PI_2;
        let dir = -normal.x().signum();

        const SWING_WIDTH: f32 = 2.0;
        let swing = SWING_WIDTH / 4.0 * dir;
        let hand_pos = center + normal / 2.0;
        #[rustfmt::skip]
        let frames = [
            ( 2.50, rot - swing * 1.0, Some(hand_pos) , false ), // ready   
            ( 2.65, rot - swing * 2.0, None           , false ), // back up 
            ( 1.00, rot + swing * 2.0, None           , true  ), // swing   
            ( 2.85, rot + swing * 3.0, None           , false ), // recovery
            ( 2.50, start_rot.0      , Some(start_pos), false ), // return  
        ];

        let (st, et, t) = (start_tick as f32, end_tick as f32, tick as f32);
        let dt = et - st;
        let total = frames.iter().map(|&(t, ..)| t).sum::<f32>();

        let mut last_tick = st;
        let mut last_rot = start_rot.vec2();
        let mut last_pos = start_pos;
        let mut do_damage = false;
        for &(duration, angle_rot, pos, damage) in &frames {
            let tick = last_tick + dt * (duration / total);
            let rot = Rot(angle_rot).vec2();
            let prog = math::inv_lerp(last_tick, tick, t).min(1.0).max(0.0);

            if prog > 0.0 {
                if prog < 1.0 {
                    do_damage = do_damage || damage;
                }

                self.rot.set_vec2(math::slerp(last_rot, rot, prog));
                if let Some(p) = pos {
                    *self.pos = last_pos.lerp(p, prog);
                    last_pos = p;
                }
            }

            last_rot = rot;
            last_tick = tick;
        }

        do_damage
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

        if is_mouse_button_down(MouseButton::Left) {
            *self.vel.0.x_mut() =
                self.vel.0.x().abs() * (mouse_position().0 - screen_width() / 2.0).signum();
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
            Art::Sword => -0.5,
            _ => 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Health(u32);

#[macroquad::main("rpg")]
async fn main() {
    let mut ecs = hecs::World::new();
    let mut damage_labels = DamageLabelBin::new();
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
        Contacts::default(),
        Art::Sword,
    ));
    let hero = ecs.spawn((
        Vec2::unit_y() * -5.0,
        Velocity(Vec2::zero()),
        Phys::wings(0.3, 0.21, CircleKind::Push),
        Contacts::default(),
        Art::Hero,
        EquippedWeapon(wep),
    ));

    ecs.spawn((
        Vec2::unit_x() * -4.0,
        Phys::wings(0.3, 0.21, CircleKind::Push),
        Contacts::default(),
        Art::Npc,
        Fixed
    ));

    ecs.spawn((
        Vec2::unit_x() * 4.0,
        Phys::new(&[Circle::push(0.3, vec2(0.0, 0.1)), Circle::hit(0.4, vec2(0.0, 0.4))]),
        Health(5),
        Contacts::default(),
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
                }
                Err(e) => {
                    error!("no hero!? {}", e);
                    continue;
                }
            };
            drawer.pan_cam_to = hero_pos;

            let hero_damaged = ecs
                .query_one_mut::<Weapon>(hero_wep)
                .ok()
                .map(|mut wep| {
                    wep.tick(WeaponInput {
                        start_attacking: is_mouse_button_down(MouseButton::Left),
                        target: drawer.cam.screen_to_world(mouse_position().into()),
                        wielder_pos: hero_pos,
                        wielder_vel: hero_vel,
                        tick,
                    });
                    wep.damage_ents()
                })
                .unwrap_or_default();

            for &hit in &hero_damaged {
                if let Ok(Health(hp)) = ecs.get_mut(hit).as_deref_mut() {
                    *hp -= 1;
                    damage_labels.push(DamageLabel {
                        tick,
                        hp: -1,
                        pos: hero_pos,
                    });
                    if *hp > 0 {
                        continue
                    }
                }
                or_err!(ecs.despawn(hit));
            }
        }

        drawer.draw(&ecs);
        damage_labels.draw(tick, &drawer.cam);
        fps.update();
        fps.draw();

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

    fn draw(&self) {
        draw_text(&self.average().to_string(), 0.0, 0.0, 30.0, BLACK);
    }
}

#[derive(Debug, Clone, Copy)]
struct DamageLabel {
    pos: Vec2,
    hp: i32,
    tick: u32
}

struct DamageLabelBin {
    labels: Vec<DamageLabel>,
    text: String,
}
impl DamageLabelBin {
    fn new() -> Self {
        Self {
            labels: Vec::with_capacity(1000),
            text: String::with_capacity(100),
        }
    }

    fn push(&mut self, label: DamageLabel) {
        self.labels.push(label);
    }

    fn draw(&mut self, tick: u32, cam: &Camera2D) {
        let Self { labels, text, .. } = self;

        labels.drain_filter(|l| {
            let end_tick = l.tick + 60;

            *text = l.hp.to_string();
            let (x, y) = cam.world_to_screen(l.pos).into();
            draw_text(text, x, y + (end_tick - tick) as f32, 20.0, BLUE);

            tick > end_tick
        });
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
                    gl.push_model_matrix(glam::Mat4::from_translation(glam::vec3(
                        pos.x(),
                        pos.y(),
                        0.0,
                    )));
                    if let Some(Rot(r)) = rot {
                        gl.push_model_matrix(glam::Mat4::from_rotation_z(r));
                    }

                    draw_triangle(
                        vec2(0.075, 0.0),
                        vec2(-0.075, 0.0),
                        vec2(0.00, GOLDEN_RATIO),
                        BROWN,
                    );
                    for &dir in &[-1.0, 1.0] {
                        draw_triangle(
                            vec2(0.00, GOLDEN_RATIO),
                            vec2(dir * 0.1, 0.35),
                            vec2(0.20 * dir, 1.35),
                            GRAY,
                        )
                    }
                    draw_triangle(
                        vec2(-0.1, 0.35),
                        vec2(0.1, 0.35),
                        vec2(0.00, GOLDEN_RATIO),
                        GRAY,
                    );

                    let (x, y) = vec2(-0.225, 0.400).into();
                    let (w, z) = vec2(0.225, 0.400).into();
                    draw_line(x, y, w, z, 0.135, DARKGRAY);
                    if rot.is_some() {
                        gl.pop_model_matrix();
                    }
                    gl.pop_model_matrix();
                    continue;
                }
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
