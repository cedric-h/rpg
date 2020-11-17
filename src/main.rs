#![feature(drain_filter)]
#![feature(array_map)]
#![feature(result_copied)]
use macroquad::prelude::*;

#[allow(dead_code)]
mod math;
use std::f32::consts::{FRAC_PI_2, TAU};

use megaui_macroquad::{megaui, mouse_over_ui};

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

fn float_cmp<T>(a: T, b: T, mut f: impl FnMut(T) -> f32) -> std::cmp::Ordering {
    f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Greater)
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

#[derive(Debug, Clone, Copy)]
struct Phys([Option<Circle>; 5]);
impl Phys {
    fn new(circles: &[Circle]) -> Self {
        let mut s = [None; 5];
        for (i, c) in circles.iter().copied().enumerate() {
            s[i] = Some(c);
        }
        Self(s)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Circle> {
        self.0.iter_mut().filter_map(|s| s.as_mut())
    }

    fn iter(&self) -> impl Iterator<Item = &Circle> {
        self.0.iter().filter_map(|s| s.as_ref())
    }

    fn pushfoot_bighit() -> Self {
        Self::new(&[Circle::push(0.3, vec2(0.0, 0.1)), Circle::hit(0.4, vec2(0.0, 0.4))])
    }

    fn pushfoot(r: f32) -> Self {
        Self::new(&[Circle::push(r, vec2(0.0, r))])
    }

    fn wings(r: f32, wr: f32, kind: CircleKind) -> Self {
        Self::new(&[
            Circle(r, vec2(0.0, r), kind),
            Circle(wr, vec2(-r, r), kind),
            Circle(wr, vec2(r, r), kind),
        ])
    }
}

#[derive(Debug, Default, Clone)]
struct Contacts(smallvec::SmallVec<[Hit; 5]>);
#[derive(Debug, Clone, Copy)]
struct Hit {
    normal: Vec2,
    depth: f32,
    circle_kinds: [CircleKind; 2],
    hit: hecs::Entity,
}

#[derive(Debug, Clone, Copy, Default)]
struct Rot(f32);
impl Rot {
    fn vec2(self) -> Vec2 {
        let (y, x) = self.0.sin_cos();
        vec2(x, y)
    }

    fn from_vec2(dir: Vec2) -> Self {
        Self(dir.y().atan2(dir.x()))
    }

    fn set_vec2(&mut self, v: Vec2) {
        *self = Self::from_vec2(v);
    }

    fn apply(self, v: Vec2) -> Vec2 {
        let len = v.length();
        let angle = Rot::from_vec2(v);
        Rot(angle.0 + self.0).vec2() * len
    }

    fn unapply(self, v: Vec2) -> Vec2 {
        let len = v.length();
        let angle = Rot::from_vec2(v);
        Rot(angle.0 - self.0).vec2() * len
    }
}

#[test]
fn apply_unapply() {
    for i in 0..8 {
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

        circles
            .extend(ecs.query::<(&Vec2, &Phys)>().iter().map(|(e, (&pos, &phys))| (e, pos, phys)));
        for (e0, p0, phys0) in circles.drain(..) {
            system!(ecs, e1,
                &p1   = &Vec2
                phys1 = &Phys
            {
                if e1 == e0 { continue }
                for &Circle(r0, o0, k0) in phys0.iter() {
                    for &Circle(r1, o1, k1) in phys1.iter() {
                        if !k0.hits(k1) { continue }

                        let delta = (p0 + o0) - (p1 + o1);
                        let dist = delta.length();
                        let depth = r0 + r1 - dist;
                        if depth > 0.0 {
                            collisions.push((e0, Hit {
                                hit: e1,
                                depth,
                                normal: delta.normalize(),
                                circle_kinds: [k0, k1],
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
        {
            for &Hit { depth, normal, circle_kinds, .. } in hits {
                if circle_kinds == [CircleKind::Push, CircleKind::Push] {
                    *pos += normal * depth;
                    *vel += normal * depth;
                }
            }
        });
    }
}

#[derive(Debug, Default, Clone)]
struct Bag {
    weapon: Option<hecs::Entity>,
    temp_weapon: Option<Item>,
    slots: [Option<Item>; 3],
}
impl Bag {
    fn take(&mut self, items: impl Iterator<Item = Item>) {
        for item in items {
            if self.weapon.is_none() && self.temp_weapon.is_none() {
                self.temp_weapon = Some(item);
            } else {
                let empty_slot = self.slots.iter_mut().find(|s| s.is_none());
                if let Some(slot) = empty_slot {
                    *slot = Some(item);
                } else {
                    dbg!("lmao ignoring item");
                }
            }
        }
    }
}

struct KeeperOfBags {
    temp: Vec<(hecs::Entity, Item)>,
}
impl KeeperOfBags {
    fn new() -> Self {
        Self { temp: Vec::with_capacity(100) }
    }

    fn keep(&mut self, ecs: &mut hecs::World) {
        self.temp.extend(ecs.query::<&mut Bag>().iter().filter_map(|(_, b)| {
            let w = b.temp_weapon.take()?;
            let e = ecs.reserve_entity();
            b.weapon = Some(e);
            Some((e, w))
        }));
        for (ent, weapon) in self.temp.drain(..) {
            or_err!(ecs.insert(ent, weapon));
        }
    }
}

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
    toward: Vec2,
}

#[derive(Debug, Clone, Copy)]
struct WeaponInput {
    start_attacking: bool,
    tick: u32,
    wielder_vel: Vec2,
    wielder_pos: Vec2,
    wielder_dir: Direction,
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
        for Circle(_, o, _) in self.phy.iter_mut() {
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
        for Circle(_, o, _) in self.phy.iter_mut() {
            *o = self.rot.apply(*o);
        }
    }

    fn ent_hits(&mut self) -> smallvec::SmallVec<[hecs::Entity; 5]> {
        match self.wep.attack {
            Attack::Swing(s) if s.doing_damage => self
                .contacts
                .0
                .iter()
                .filter(|h| matches!(h.circle_kinds[0], CircleKind::Hurt | CircleKind::Hit))
                .map(|h| h.hit)
                .filter(|e| {
                    if self.wep.ents_hit.contains(e) {
                        false
                    } else {
                        self.wep.ents_hit.push(*e);
                        true
                    }
                })
                .collect(),
            _ => Default::default(),
        }
    }

    fn rest(
        &mut self,
        WeaponInput { wielder_pos, wielder_vel, wielder_dir, tick, .. }: WeaponInput,
    ) -> (Rot, Vec2) {
        let dir = wielder_dir.signum();
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
        let WeaponInput { wielder_pos, start_attacking, target, tick, .. } = input;

        use Attack::*;
        self.wep.attack = match self.wep.attack {
            Ready if start_attacking => {
                self.wep.ents_hit.clear();
                Swing(SwingState {
                    start_tick: tick,
                    end_tick: tick + 50,
                    toward: self.from_center(wielder_pos, target),
                    doing_damage: false,
                })
            }
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
        SwingState { start_tick, end_tick, toward, .. }: SwingState,
    ) -> bool {
        let WeaponInput { wielder_pos, tick, .. } = input;
        let (start_rot, start_pos) = self.rest(input);

        let center = self.center(wielder_pos);
        let rot = Rot::from_vec2(toward).0 - FRAC_PI_2;
        let dir = -toward.x().signum();

        const SWING_WIDTH: f32 = 2.0;
        let swing = SWING_WIDTH / 4.0 * dir;
        let hand_pos = center + toward / 2.0;
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

#[derive(Copy, Clone, Debug)]
enum Direction {
    Left,
    Right,
}
impl Direction {
    fn signum(self) -> f32 {
        match self {
            Direction::Left => -1.0,
            Direction::Right => 1.0,
        }
    }
}
impl From<f32> for Direction {
    fn from(f: f32) -> Self {
        if f < 0.0 {
            Direction::Left
        } else {
            Direction::Right
        }
    }
}

#[derive(hecs::Query, Debug)]
struct Hero<'a> {
    vel: &'a mut Velocity,
    pos: &'a mut Vec2,
    art: &'a Art,
    dir: &'a mut Direction,
    bag: &'a mut Bag,
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
            self.vel.0 += move_vec
                * if self.bag.weapon.is_none()
                    || move_vec.x() == 0.0
                    || move_vec.x().signum() == self.dir.signum()
                {
                    0.0075
                } else {
                    0.0045
                };
        }

        if !mouse_over_ui() && is_mouse_button_down(MouseButton::Left) {
            if mouse_position().0 < screen_width() / 2.0 {
                *self.dir = Direction::Left;
            } else {
                *self.dir = Direction::Right;
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Art {
    Hero,
    Scarecrow,
    TripletuftedTerrorworm,
    Npc,
    Sword,
    Post,
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
/// Entities with Innocence may be impervious to damage.
enum Innocence {
    Unwavering,
    /// Innocence can expire after a set date.
    Expires(u32),
}
impl Innocence {
    fn active(self, tick: u32) -> bool {
        match self {
            Innocence::Unwavering => true,
            Innocence::Expires(on) => tick <= on,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Health(u32);

#[derive(hecs::Bundle, Clone, Debug)]
struct Item {
    pos: Vec2,
    art: Art,
    rot: Rot,
    phys: Phys,
    wep: WeaponState,
    contacts: Contacts,
}
impl Item {
    fn sword() -> Self {
        Self {
            pos: Vec2::zero(),
            art: Art::Sword,
            rot: Rot(0.0),
            phys: Phys::new(&[
                Circle::hurt(0.2, vec2(0.0, 1.35)),
                Circle::hurt(0.185, vec2(0.0, 1.1)),
                Circle::hurt(0.15, vec2(0.0, 0.85)),
                Circle::hurt(0.125, vec2(0.0, 0.65)),
            ]),
            wep: WeaponState::default(),
            contacts: Contacts::default(),
        }
    }
}

fn rand_vec2() -> Vec2 {
    Rot(rand::rand() as f32 / (u32::MAX as f32 / TAU)).vec2()
}

struct Wander {
    pen_pos: Vec2,
    pen_radius: f32,
    speed: f32,
    goal: Vec2,
}
impl Wander {
    fn around(pen_pos: Vec2, pen_radius: f32) -> Self {
        Wander {
            pen_pos,
            pen_radius,
            speed: 0.001,
            goal: rand_vec2() * rand::gen_range(0.0, pen_radius) + pen_pos,
        }
    }
}

fn wander(ecs: &mut hecs::World) {
    system!(ecs, _,
        w           = &mut Wander
        Velocity(v) = &mut _
        &p          = &Vec2
    {
        let delta = w.goal - p;
        if delta.length() < 0.05 {
            w.goal = rand_vec2() * rand::gen_range(0.0, w.pen_radius) + w.pen_pos;
        }
        *v += delta.normalize() * w.speed;
    });
}

fn circle_points(n: usize) -> impl Iterator<Item = Vec2> {
    (0..n).map(move |i| Rot((i as f32 / n as f32) * TAU).vec2())
}

fn update_hero(hero: hecs::Entity, game: &mut Game, quests: &mut Quests) {
    let &Game { tick, mouse_pos, .. } = &*game;

    let (hero_pos, hero_wep, hero_vel, hero_dir) = match game.ecs.query_one_mut::<Hero>(hero) {
        Ok(mut hero) => {
            hero.bag.take(quests.hero_rewards.drain(..));
            hero.movement();
            (*hero.pos, hero.bag.weapon, hero.vel.0, *hero.dir)
        }
        Err(e) => {
            error!("no hero!? {}", e);
            return;
        }
    };
    game.hero_pos = hero_pos;

    let hero_attacks = hero_wep
        .and_then(|e| game.ecs.query_one_mut::<Weapon>(e).ok())
        .map(|mut wep| {
            wep.tick(WeaponInput {
                start_attacking: !mouse_over_ui() && is_mouse_button_down(MouseButton::Left),
                target: mouse_pos,
                wielder_pos: hero_pos,
                wielder_vel: hero_vel,
                wielder_dir: hero_dir,
                tick,
            });
            wep.ent_hits()
        })
        .unwrap_or_default();

    for &hit in &hero_attacks {
        if game.hit(hit) {
            quests.update(game);
        }
    }
}

struct Enemy;
struct Waffle {
    enemies: Vec<(hecs::Entity, Vec2, Velocity)>,
}
impl Waffle {
    fn new() -> Self {
        Self { enemies: Vec::with_capacity(100) }
    }

    fn update(&mut self, game: &mut Game) {
        let &mut Game { hero_pos, tick, .. } = game;
        let Self { enemies, .. } = self;
        enemies
            .extend(game.ecs.query::<(&_, &_, &Enemy)>().iter().map(|(e, (x, v, _))| (e, *x, *v)));
        enemies.sort_by(|a, b| float_cmp(a, b, |&(_, p, _)| (hero_pos - p).length_squared()));

        const SLOT_COUNT: usize = 7;
        let mut slots = [false; SLOT_COUNT];
        for (e, pos, vel) in enemies.drain(..) {
            let slot = circle_points(SLOT_COUNT)
                .map(|v| (Rot(0.1).apply(v) * 1.6 + hero_pos) - pos)
                .enumerate()
                .filter(|&(i, _)| !slots[i])
                .min_by(|a, b| float_cmp(a, b, |d| d.1.length_squared()));

            if let Some((i, delta)) = slot {
                slots[i] = true;
                if let Ok(Velocity(vel)) = game.ecs.get_mut(e).as_deref_mut() {
                    *vel += delta * 0.0035;
                }
            }

            if let Ok(Some(wep_ent)) = game.ecs.get::<Bag>(e).map(|b| b.weapon) {
                let attacks = game
                    .ecs
                    .query_one_mut::<Weapon>(wep_ent)
                    .ok()
                    .map(|mut wep| {
                        wep.tick(WeaponInput {
                            start_attacking: false,
                            target: hero_pos,
                            wielder_pos: pos,
                            wielder_vel: vel.0,
                            wielder_dir: (hero_pos - pos).x().into(),
                            tick,
                        });
                        wep.ent_hits()
                    })
                    .unwrap_or_default();

                for &hit in &attacks {
                    game.hit(hit);
                }
            }
        }
    }
}

#[macroquad::main("rpg")]
async fn main() {
    let mut ecs = hecs::World::new();
    let mut drawer = Drawer::new();
    let mut waffle = Waffle::new();
    let mut physics = Physics::new();
    let mut quests = Quests::new();
    let mut keeper_of_bags = KeeperOfBags::new();
    let mut fps = Fps::new();

    let npc = |npc_pos| (npc_pos, Phys::wings(0.3, 0.21, CircleKind::Push), Art::Npc);

    let hero = ecs.spawn((
        Vec2::unit_x() * -2.5,
        Velocity(Vec2::zero()),
        Phys::wings(0.3, 0.21, CircleKind::Push),
        Art::Hero,
        Direction::Left,
        Bag::default(),
    ));

    type Post = (Vec2, Art, Phys);
    #[derive(Clone, Copy)]
    struct Pen {
        gate: [(hecs::Entity, Post); 2],
        pos: Vec2,
        radius: f32,
    }
    impl Pen {
        fn new(ecs: &mut hecs::World) -> Self {
            let mut pen = Self {
                gate: unsafe { [std::mem::zeroed(), std::mem::zeroed()] },
                radius: 4.5,
                pos: vec2(5.0, 5.0),
            };
            for (i, post_pos) in circle_points(36).enumerate() {
                let post = pen.post(post_pos);
                let e = ecs.spawn(post.clone());

                let gate_i = i - 36 / 4;
                if gate_i <= 1 {
                    pen.gate[gate_i] = (e, post);
                }
            }

            pen
        }

        fn post(&self, post_pos: Vec2) -> Post {
            (post_pos * self.radius + self.pos, Art::Post, Phys::pushfoot(0.4))
        }

        fn gate_open(&self, ecs: &mut hecs::World) {
            for (e, _) in self.gate.iter() {
                or_err!(ecs.remove::<(Phys, Art)>(*e));
            }
        }

        fn gate_close(&self, ecs: &mut hecs::World) {
            for (e, c) in self.gate.iter() {
                or_err!(ecs.insert(*e, c.clone()));
            }
        }
    }

    let npc2 = ecs.spawn(npc(vec2(4.0, 11.0)));
    let pen = Pen::new(&mut ecs);
    let (pen_pos, pen_radius) = (pen.pos, pen.radius);
    let terrorworms = {
        let mut circle = circle_points(6);
        [(); 6].map(|_| {
            ecs.spawn((
                circle.next().unwrap() * 2.0 + pen.pos,
                Art::TripletuftedTerrorworm,
                Wander::around(pen.pos, pen.radius * 0.4),
                Velocity(Vec2::zero()),
                Innocence::Unwavering,
                Phys::pushfoot_bighit(),
                Health(1),
            ))
        })
    };

    let rpg_tropes_4 = quests.add(Quest {
        title: "RPG Tropes IV - Tripletufted Terrorworms",
        completion_quip: "Of course they just come back ... you need a Honeycoated Heptahorn!",
        unlock_description: concat!(
            "Oh, adventurer, I'm so glad you've come along! \n",
            "In fact, it's almost like I've just been standing here since \n",
            "the dawn of time itself, waiting for someone to come help me. In fact, \n",
            "it's almost as if I was created simply to give you something to do. \n",
            "In this pen are the last Tripletufted Terrorworms known to exist, and \n",
            "I need you to forgo your conscience and simply slaughter them: no remorse.",
        ),
        on_accept: Some(Box::new(move |g| pen.gate_open(&mut g.ecs))),
        tasks: vec![
            Task {
                label: "Enter the Pen",
                req: Box::new(move |g| g.hero_dist(pen_pos) < pen_radius * 0.9),
                on_finish: Box::new(move |g| {
                    for &tw in &terrorworms {
                        drop(g.innocent_for(tw, 1));
                    }
                    pen.gate_close(&mut g.ecs)
                }),
                ..Default::default()
            },
            Task {
                label: "Slaughter the first one",
                req: Box::new(move |g| terrorworms.iter().filter(|&&e| g.dead(e)).count() == 1),
                on_finish: Box::new(move |g| {
                    for &tw in &terrorworms {
                        drop(g.innocent_for(tw, 10));
                        if let Ok(Wander { goal, speed, .. }) = g.ecs.get_mut(tw).as_deref_mut() {
                            *goal = rand_vec2() * pen_radius + pen_pos;
                            *speed *= 2.5;
                        }
                    }
                }),
                ..Default::default()
            },
            Task {
                label: "Slaughter the second one",
                req: Box::new(move |g| terrorworms.iter().filter(|&&e| g.dead(e)).count() == 2),
                on_finish: Box::new(move |g| {
                    for &tw in &terrorworms {
                        drop(g.innocent_for(tw, 10));
                        drop(g.ecs.remove_one::<Wander>(tw));
                        let mut bag = Bag::default();
                        bag.take(std::iter::once(Item::sword()));
                        drop(g.ecs.insert(tw, (Enemy, bag)));
                    }
                }),
                ..Default::default()
            },
        ],
        ..Default::default()
    });

    let rpg_tropes_3 = quests.add(Quest {
        title: "RPG Tropes III - Couriers",
        completion_quip: "Save us from impending doom! But first, give this to my girlfriend.",
        unlock_description: concat!(
            "I'm happy to say that you've come to the part of your journey where \n",
            "I must bid you adieu and direct you to another clueless, doddering NPC. \n",
            "So long adventurer, and thanks for nothing! \n\n",
            "P.S. Don't forget to save me from certain doom!",
        ),
        tasks: vec![Task {
            label: "Talk to NPC 2",
            req: Box::new(move |g| g.hero_dist_ent(npc2) < 1.3),
            ..Default::default()
        }],
        unlocks: vec![rpg_tropes_4],
        ..Default::default()
    });

    ecs.spawn(npc(vec2(-4.0, 0.0)));
    let scarecrow =
        ecs.spawn((vec2(3.4, -0.4), Phys::pushfoot_bighit(), Health(5), Art::Scarecrow));
    let rpg_tropes_2 = quests.add(Quest {
        title: "RPG Tropes II - Scarecrows",
        completion_quip: "My wife and children -- who probably don't even exist -- thank you!",
        unlock_description: concat!(
            "I'm going to have to ask you to ruthlessly destroy a scarecrow I \n",
            "painstakingly made from scratch over the course of several days \n",
            "because this game takes place before modern manufacturing practices, \n",
            "but it's fine because it's not like I have anything else to do other \n",
            "than stand here pretending to work on another one!",
        ),
        tasks: vec![Task {
            label: "Destroy Scarecrow",
            req: Box::new(move |g| g.dead(scarecrow)),
            ..Default::default()
        }],
        unlocks: vec![rpg_tropes_3],
        ..Default::default()
    });

    quests.add(Quest {
        title: "RPG Tropes Abound!",
        completion_quip: "Try not to poke your eye out, like the last adventurer ...",
        unlock_description: concat!(
            "It's dangerous to go alone! \n",
            "Allow me to give you, a complete stranger, a dangerous weapon \n",
            "because I just happen to believe that you may be capable and willing \n",
            "to spare us from the horrible fate that will surely befall us regardless \n",
            "of whether or not you spend three hours immersed in a fishing minigame.",
        ),
        unlocks: vec![rpg_tropes_2],
        reward_items: vec![Item::sword()],
        completion: QuestCompletion::Unlocked,
        ..Default::default()
    });

    const STEP_EVERY: f64 = 1.0 / 60.0;
    let mut time = STEP_EVERY;
    let mut step = get_time();
    let mut game = Game { ecs, hero_pos: Vec2::zero(), tick: 0, mouse_pos: Vec2::zero() };
    loop {
        time += get_time() - step;
        step = get_time();
        while time >= STEP_EVERY {
            time -= STEP_EVERY;
            game.tick = game.tick.wrapping_add(1);
            game.mouse_pos = drawer.cam.screen_to_world(mouse_position().into());

            physics.tick(&mut game.ecs);
            quests.update(&mut game);
            waffle.update(&mut game);
            keeper_of_bags.keep(&mut game.ecs);
            wander(&mut game.ecs);

            update_hero(hero, &mut game, &mut quests);

            drawer.update(&mut game);
        }

        drawer.draw(&game);

        quests.ui(&game);
        megaui_macroquad::draw_megaui();

        fps.update();
        fps.draw();

        next_frame().await
    }
}

struct Game {
    ecs: hecs::World,
    hero_pos: Vec2,
    /// In game coordinates
    mouse_pos: Vec2,
    tick: u32,
}
impl Game {
    fn dead(&self, e: hecs::Entity) -> bool {
        !self.ecs.contains(e)
    }

    fn hero_dist(&self, p: Vec2) -> f32 {
        (p - self.hero_pos).length()
    }

    fn hero_dist_ent(&self, e: hecs::Entity) -> f32 {
        self.ecs.get::<Vec2>(e).map(|p| (*p - self.hero_pos).length()).unwrap_or(f32::INFINITY)
    }

    fn innocent_for(&mut self, e: hecs::Entity, duration: u32) -> Result<(), hecs::NoSuchEntity> {
        let ends = self.tick + duration;
        self.ecs.insert_one(e, Innocence::Expires(ends))
    }

    /// Returns true if they died
    fn hit(&mut self, hit: hecs::Entity) -> bool {
        let tick = self.tick;

        if let Ok((Health(hp), &pos, innocence)) =
            self.ecs.query_one_mut::<(&mut _, &_, Option<&Innocence>)>(hit)
        {
            if let Some(true) = innocence.map(|i| i.active(tick)) {
                return false;
            }

            *hp -= 1;
            let dead = *hp == 0;
            if dead {
                or_err!(self.ecs.despawn(hit));
            }
            self.ecs.spawn((DamageLabel { tick, hp: -1, pos },));

            dead
        } else {
            false
        }
    }
}

struct Task {
    label: &'static str,
    req: Box<dyn FnMut(&Game) -> bool>,
    on_finish: Box<dyn FnMut(&mut Game)>,
    finished: bool,
}
impl Default for Task {
    fn default() -> Self {
        Self { label: "", req: Box::new(|_| true), on_finish: Box::new(|_| {}), finished: false }
    }
}
impl Task {
    fn done(&mut self, g: &mut Game) -> bool {
        let Self { finished, req, .. } = self;
        if !*finished {
            *finished = (*req)(&*g);
            if *finished {
                (self.on_finish)(g);
            }
        }
        *finished
    }
}

#[derive(Default)]
struct Quest {
    title: &'static str,
    unlock_description: &'static str,
    completion_quip: &'static str,
    on_accept: Option<Box<dyn FnMut(&mut Game)>>,
    tasks: Vec<Task>,
    unlocks: Vec<usize>,
    reward_items: Vec<Item>,
    completion: QuestCompletion,
}

#[derive(Debug, Clone, Copy)]
enum QuestCompletion {
    Locked,
    Unlocked,
    Accepted { on_tick: u32 },
    Finished { on_tick: u32 },
}
impl Default for QuestCompletion {
    fn default() -> Self {
        QuestCompletion::Locked
    }
}
impl QuestCompletion {
    fn locked(self) -> bool {
        matches!(self, QuestCompletion::Locked)
    }

    fn unlocked(self) -> bool {
        matches!(self, QuestCompletion::Unlocked)
    }

    fn unlock(&mut self) {
        if self.locked() {
            *self = QuestCompletion::Unlocked;
        }
    }

    fn accepted(self) -> Option<u32> {
        if let QuestCompletion::Accepted { on_tick } = self {
            Some(on_tick)
        } else {
            None
        }
    }

    fn accept(&mut self, on_tick: u32) {
        if self.unlocked() {
            *self = QuestCompletion::Accepted { on_tick };
        }
    }

    fn finished(self) -> Option<u32> {
        if let QuestCompletion::Finished { on_tick } = self {
            Some(on_tick)
        } else {
            None
        }
    }

    fn finish(&mut self, on_tick: u32) {
        if self.accepted().is_some() {
            *self = QuestCompletion::Finished { on_tick };
        }
    }
}

struct QuestVec(Vec<Quest>);
impl QuestVec {
    fn unlocked_mut(&mut self) -> impl Iterator<Item = (usize, &mut Quest)> {
        self.0.iter_mut().enumerate().filter(|(_, q)| q.completion.unlocked())
    }

    fn accepted_mut(&mut self) -> impl Iterator<Item = (u32, usize, &mut Quest)> {
        self.0.iter_mut().enumerate().filter_map(|(i, q)| Some((q.completion.accepted()?, i, q)))
    }

    fn finished_mut(&mut self) -> impl Iterator<Item = (u32, usize, &mut Quest)> {
        self.0.iter_mut().enumerate().filter_map(|(i, q)| Some((q.completion.finished()?, i, q)))
    }

    fn unlocked(&self) -> impl Iterator<Item = (usize, &Quest)> {
        self.0.iter().enumerate().filter(|(_, q)| q.completion.unlocked())
    }

    fn accepted(&self) -> impl Iterator<Item = (u32, usize, &Quest)> {
        self.0.iter().enumerate().filter_map(|(i, q)| Some((q.completion.accepted()?, i, q)))
    }

    fn finished(&self) -> impl Iterator<Item = (u32, usize, &Quest)> {
        self.0.iter().enumerate().filter_map(|(i, q)| Some((q.completion.finished()?, i, q)))
    }
}

pub fn open_tree<F: FnOnce(&mut megaui::Ui)>(
    ui: &mut megaui::Ui,
    id: megaui::Id,
    label: &str,
    f: F,
) -> bool {
    megaui::widgets::TreeNode::new(id, label).init_unfolded().ui(ui, f)
}

struct Quests {
    quests: QuestVec,
    hero_rewards: Vec<Item>,
    temp: Vec<usize>,
    tab_titles: [String; 3],
    new_tabs: [bool; 3],
    jump_to_tab: Option<usize>,
}
impl Quests {
    fn new() -> Self {
        Self {
            quests: QuestVec(Vec::with_capacity(100)),
            hero_rewards: Vec::with_capacity(100),
            temp: Vec::with_capacity(100),
            tab_titles: [(); 3].map(|_| String::with_capacity(25)),
            new_tabs: [false; 3],
            jump_to_tab: None,
        }
    }

    fn add(&mut self, q: Quest) -> usize {
        let i = self.quests.0.len();
        self.quests.0.push(q);
        i
    }

    fn update(&mut self, g: &mut Game) {
        for (tick, _, quest) in self.quests.accepted_mut() {
            if tick + 1 == g.tick {
                if let Some(f) = &mut quest.on_accept {
                    (f)(g);
                }
            }

            if quest.tasks.iter_mut().all(|t| t.done(g)) {
                quest.completion.finish(g.tick);
                self.jump_to_tab = Some(2);
                self.temp.extend(quest.unlocks.iter().copied());
                self.hero_rewards.extend(quest.reward_items.iter().cloned());
            }
        }

        for unlock in self.temp.drain(..) {
            self.new_tabs[0] = true;
            self.quests.0[unlock].completion.unlock();
        }
    }

    fn unlocked_ui(&mut self, ui: &mut megaui::Ui, tick: u32) {
        use megaui::widgets::Label;

        ui.separator();
        for (_, quest) in self.quests.unlocked_mut() {
            ui.label(None, quest.title);
            Label::new(quest.unlock_description).multiline(14.0).ui(ui);
            if ui.button(None, "Accept") {
                quest.completion.accept(tick);
                self.new_tabs[1] = true;
            }
            ui.separator();
        }
    }

    fn accepted_ui(&mut self, ui: &mut megaui::Ui) {
        use megaui::hash;

        ui.separator();
        for (_, i, quest) in self.quests.accepted() {
            open_tree(ui, hash!(i), quest.title, |ui| {
                for task in &quest.tasks {
                    ui.label(None, &task.label);
                }
            });
            ui.separator();
        }
    }

    fn finished_ui(&mut self, ui: &mut megaui::Ui) {
        use megaui::{hash, widgets::Label};

        ui.separator();
        for (i, _, quest) in self.quests.finished() {
            open_tree(ui, hash!(i), quest.title, |ui| {
                ui.label(None, quest.completion_quip);
                open_tree(ui, hash!(i, "r"), "Rewards", |ui| {
                    for item in &quest.reward_items {
                        ui.label(None, &format!("{:#?}", item.art));
                    }
                });
                ui.tree_node(hash!(i, "u"), "Unlock Text", |ui| {
                    Label::new(quest.unlock_description).multiline(14.0).ui(ui);
                });
            });
            ui.separator();
        }
    }

    fn size() -> Vec2 {
        vec2(520.0, 230.0)
    }

    fn tab_titles(&mut self) -> [&str; 3] {
        for (i, &n) in [
            self.quests.unlocked().count(),
            self.quests.accepted().count(),
            self.quests.finished().count(),
        ]
        .iter()
        .enumerate()
        {
            self.tab_titles[i] = format!(
                "{} [{}] {}",
                if self.new_tabs[i] { "NEW! " } else { "" },
                n,
                ["Unlocked", "Accepted", "Finished"][i],
            );
        }
        [&self.tab_titles[0], &self.tab_titles[1], &self.tab_titles[2]]
    }

    fn ui(&mut self, g: &Game) {
        use megaui_macroquad::{
            draw_window,
            megaui::{
                hash,
                widgets::{Group, Tabbar},
                Vector2,
            },
            WindowParams,
        };

        let size = Self::size();
        draw_window(
            hash!(),
            vec2(screen_width(), screen_height()) / 2.0 - size * vec2(0.5, -0.1),
            size,
            WindowParams { label: "Quests".to_string(), ..Default::default() },
            |ui| {
                let tab_height = 22.5;
                let tab = {
                    let jump = self.jump_to_tab.take();
                    let titles = self.tab_titles();
                    let mut tabbar = Tabbar::new(
                        hash!(),
                        Vector2::new(0.0, 0.0),
                        Vector2::new(size.x(), tab_height),
                        &titles,
                    );
                    if let Some(n) = jump {
                        tabbar = tabbar.jump_to_tab(n);
                    }
                    tabbar.ui(ui)
                };

                self.new_tabs[tab as usize] = false;
                Group::new(hash!(), Vector2::new(size.x(), size.y() - tab_height * 2.0))
                    .position(Vector2::new(0.0, tab_height))
                    .ui(ui, |ui| match tab {
                        0 => self.unlocked_ui(ui, g.tick),
                        1 => self.accepted_ui(ui),
                        2 => self.finished_ui(ui),
                        _ => unreachable!(),
                    });
            },
        );
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
    tick: u32,
}
impl DamageLabel {
    fn end_tick(self) -> u32 {
        self.tick + 60
    }
}

struct DamageLabelBin {
    stale: Vec<hecs::Entity>,
    text: String,
}
impl DamageLabelBin {
    fn new() -> Self {
        Self { stale: Vec::with_capacity(100), text: String::with_capacity(100) }
    }

    fn update(&mut self, Game { ecs, tick, .. }: &mut Game) {
        self.stale.extend(
            ecs.query::<&DamageLabel>()
                .iter()
                .filter(|(_, dl)| *tick > dl.end_tick())
                .map(|(e, _)| e),
        );
        for e in self.stale.drain(..) {
            or_err!(ecs.despawn(e));
        }
    }

    fn draw(&mut self, Game { ecs, tick: game_tick, .. }: &Game, cam: &Camera2D) {
        let Self { text, .. } = self;

        for (_, &DamageLabel { hp, pos, tick }) in ecs.query::<&_>().iter() {
            *text = hp.to_string();
            let (x, y) = cam.world_to_screen(pos).into();
            draw_text(text, x - 20.0, y - 120.0 - (game_tick - tick) as f32, 42.0, MAROON);
        }
    }
}

struct Drawer {
    sprites: Vec<(Vec2, Art, Option<Rot>)>,
    damage_labels: DamageLabelBin,
    cam: Camera2D,
}
impl Drawer {
    fn new() -> Self {
        Self {
            sprites: Vec::with_capacity(1000),
            cam: Default::default(),
            damage_labels: DamageLabelBin::new(),
        }
    }

    fn update(&mut self, game: &mut Game) {
        let Self { damage_labels, cam, .. } = self;
        damage_labels.update(game);
        cam.target = cam.target.lerp(game.hero_pos, 0.05);
    }

    fn draw(&mut self, game: &Game) {
        self.sprites(game);
        self.damage_labels.draw(game, &self.cam);
    }

    fn sprites(&mut self, Game { ecs, .. }: &Game) {
        let Self { sprites, cam, .. } = self;
        cam.zoom = vec2(1.0, screen_width() / screen_height()) / 7.8;
        set_camera(*cam);

        clear_background(Color([180, 227, 245, 255]));

        let gl = unsafe { get_internal_gl().quad_gl };

        sprites.extend(
            ecs.query::<(&_, &_, Option<&Rot>)>().iter().map(|(_, (&p, &a, r))| (p, a, r.copied())),
        );
        sprites.sort_by(|a, b| float_cmp(b, a, |(pos, art, _)| pos.y() + art.z_offset()));
        for (pos, art, rot) in sprites.drain(..) {
            const GOLDEN_RATIO: f32 = 1.618034;

            let (color, w, h) = match art {
                Art::Hero => (BLUE, 1.0, 1.0),
                Art::Scarecrow => (RED, 0.8, 0.8),
                Art::TripletuftedTerrorworm => (WHITE, 0.8, 0.8),
                Art::Npc => (GREEN, 1.0, GOLDEN_RATIO),
                Art::Post => (BROWN, 0.8, GOLDEN_RATIO * 0.8),
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
            phys = &Phys
            &pos = &Vec2
        {
            use CircleKind::*;

            fn kind_color(kind: CircleKind) -> Color {
                match kind {
                    Push => PURPLE,
                    Hit => PINK,
                    Hurt => DARKBLUE,
                }
            }

            for &Circle(r, o, k) in phys.iter() {
                let (x, y) = (pos + o).into();
                let mut color = kind_color(k);
                color.0[3] = 100;
                draw_circle(x, y, r, color);
            }

            fn centroid(phys: &Phys, kind: CircleKind) -> Vec2 {
                let (count, sum) = phys
                    .iter()
                    .filter(|&&Circle(_, _, k)| k == kind)
                    .fold((0, Vec2::zero()), |(i, acc), &Circle(_, p, ..)| (i + 1, acc + p));
                sum / count as f32
            }

            for &kind in &[Push, Hit, Hurt] {
                let (x, y) = (pos + centroid(phys, kind)).into();
                draw_circle(x, y, 0.1, kind_color(kind));
            }
        });

        set_default_camera();
    }
}
