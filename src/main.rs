#![feature(drain_filter)]
#![feature(array_map)]
#![feature(result_copied)]
use macroquad::prelude::*;

#[allow(dead_code)]
mod math;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use megaui_macroquad::{
    megaui::{self, hash},
    mouse_over_ui,
};

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
struct MinVelocity(Vec2);

#[derive(Debug, Copy, Clone, Default)]
struct Velocity(Vec2);
impl Velocity {
    fn knockback(&mut self, delta: Vec2, force: f32) {
        if delta.length_squared() > 0.0 {
            self.0 += delta.normalize() * force;
        }
    }
}

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
struct Phys([Option<Circle>; 6]);
impl Phys {
    fn new() -> Self {
        Self([None; 6])
    }

    fn with(circles: &[Circle]) -> Self {
        let mut s = Self::new();
        for c in circles.iter().copied() {
            s = s.insert(c);
        }
        s
    }

    fn insert(mut self, c: Circle) -> Self {
        for slot in self.0.iter_mut() {
            if slot.is_none() {
                *slot = Some(c);
                return self;
            }
        }
        panic!("no more than five circles");
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Circle> {
        self.0.iter_mut().filter_map(|s| s.as_mut())
    }

    fn iter(&self) -> impl Iterator<Item = &Circle> {
        self.0.iter().filter_map(|s| s.as_ref())
    }

    fn pushfoot_bighit(self) -> Self {
        self.insert(Circle::push(0.3, vec2(0.0, 0.1))).insert(Circle::hit(0.4, vec2(0.0, 0.4)))
    }

    fn wings(self, r: f32, wr: f32, kind: CircleKind) -> Self {
        self.insert(Circle(r, vec2(0.0, r), kind))
            .insert(Circle(wr, vec2(-r, r), kind))
            .insert(Circle(wr, vec2(r, r), kind))
    }

    fn pushbox(self, r: f32) -> Self {
        self.insert(Circle::push(r, vec2(0.0, r)))
    }

    #[allow(dead_code)]
    fn hurtbox(self, r: f32) -> Self {
        self.insert(Circle::hurt(r, vec2(0.0, r)))
    }

    fn hitbox(self, r: f32) -> Self {
        self.insert(Circle::hit(r, vec2(0.0, r)))
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
struct Scale(Vec2);

#[derive(Debug, Clone, Copy, Default)]
struct ZOffset(f32);

#[derive(Debug, Clone, Copy)]
struct Opaqueness(u8);
impl Default for Opaqueness {
    fn default() -> Self {
        Opaqueness(255)
    }
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
            min_vel       = Option<&MinVelocity>
        {
            if let Some(&MinVelocity(mv)) = min_vel {
                *pos += vel.signum() * vel.abs().max(mv.abs());
            } else {
                *pos += *vel;
            }
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

#[derive(Debug, Clone)]
enum BagWeapon {
    Out { ent: hecs::Entity, item: WeaponItem, wants_in: bool },
    In { item: WeaponItem, wants_out: bool },
}
impl BagWeapon {
    fn out(&mut self) -> Option<hecs::Entity> {
        match self {
            &mut BagWeapon::Out { ent, .. } => Some(ent),
            BagWeapon::In { wants_out, .. } => {
                *wants_out = true;
                None
            }
        }
    }

    fn is_in(&self) -> bool {
        match self {
            BagWeapon::Out { .. } => false,
            BagWeapon::In { .. } => true,
        }
    }

    fn get_in(&mut self) {
        match self {
            BagWeapon::Out { wants_in, .. } => *wants_in = true,
            BagWeapon::In { .. } => {}
        }
    }

    fn item(&self) -> &WeaponItem {
        match self {
            BagWeapon::Out { item, .. } => &item,
            BagWeapon::In { item, .. } => &item,
        }
    }

    fn wants_out(&self) -> bool {
        match self {
            BagWeapon::Out { .. } => false,
            BagWeapon::In { wants_out, .. } => *wants_out,
        }
    }

    fn wants_in(&self) -> bool {
        match self {
            BagWeapon::Out { wants_in, .. } => *wants_in,
            BagWeapon::In { .. } => false,
        }
    }

    fn ent(&self) -> Option<hecs::Entity> {
        match self {
            BagWeapon::Out { ent, .. } => Some(*ent),
            BagWeapon::In { .. } => None,
        }
    }

    fn take_item(self) -> Option<WeaponItem> {
        match self {
            BagWeapon::Out { .. } => None,
            BagWeapon::In { item, .. } => Some(item),
        }
    }
}

const BAG_SLOTS: usize = 12;
const BAG_TRINKETS: usize = 4;
#[derive(Debug, Default, Clone)]
struct Bag {
    weapon: Option<BagWeapon>,
    slots: [Option<Item>; BAG_SLOTS],
    trinkets: [Option<Trinket>; BAG_TRINKETS],
    new_item: bool,
}
impl Bag {
    fn holding(item: Item) -> Self {
        let mut me = Self::default();
        me.take(item);
        me
    }

    fn take(&mut self, item: Item) {
        match item {
            Item::Weapon(wep) if self.weapon.is_none() => {
                self.weapon = Some(BagWeapon::In { item: wep, wants_out: false });
                self.new_item = true;
                return;
            }
            Item::Trinket(t) if self.trinket_space().is_some() => {
                self.trinkets[self.trinket_space().unwrap()] = Some(t);
                self.new_item = true;
                return;
            }
            _ => {}
        }

        let empty_slot = self.slots.iter_mut().find(|s| s.is_none());
        if let Some(slot) = empty_slot {
            *slot = Some(item);
            self.new_item = true;
        } else {
            dbg!("lmao ignoring item");
        }
    }

    fn trinket_space(&self) -> Option<usize> {
        self.trinkets.iter().enumerate().find(|(_, t)| t.is_none()).map(|(i, _)| i)
    }

    fn mods(&self) -> HeroMod {
        self.trinkets.iter().filter_map(|t| *t).fold(HeroMod::default(), |a, t| a + t.mods())
    }
}

#[derive(Debug)]
struct ItemMove {
    from: SlotHandle,
    to: SlotHandle,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
enum SlotHandle {
    Weapon,
    Trinket(usize),
    Loose(usize),
}

struct BagUi {
    moves: Vec<ItemMove>,
    stomach: Vec<Consumable>,

    handle_bank: fxhash::FxHashMap<u64, SlotHandle>,

    selected: Option<SlotHandle>,

    window_open: bool,
    dragging: bool,

    icon_ent: hecs::Entity,
    icon_jumping: bool,
}
impl BagUi {
    fn new(ecs: &mut hecs::World) -> Self {
        Self {
            moves: Vec::with_capacity(100),
            stomach: Vec::with_capacity(100),
            handle_bank: std::iter::once(SlotHandle::Weapon)
                .chain((0..BAG_SLOTS).map(|i| SlotHandle::Loose(i)))
                .chain((0..BAG_TRINKETS).map(|i| SlotHandle::Trinket(i)))
                .map(|i| (hash!(i), i))
                .collect(),
            icon_ent: ecs.spawn((vec2(0.0, 0.0), Art::Lockbox, ZOffset(-999.0))),
            icon_jumping: false,
            window_open: false,
            selected: None,
            dragging: false,
        }
    }

    fn update_icon(&mut self, g: &mut Game, drawer: &Drawer) {
        if let Ok(mut pos) = g.ecs.get_mut::<Vec2>(self.icon_ent) {
            let bob =
                if self.icon_jumping { (g.tick as f32 / 8.0).sin().abs() * 0.25 } else { 0.0 };
            *pos = drawer.cam.target - drawer.screen * vec2(1.0, -1.0) + vec2(1.5, 0.1 + bob);

            if is_key_pressed(KeyCode::E)
                || g.mouse_down_tick == Some(g.tick)
                    && (*pos + vec2(0.0, 0.5) - g.mouse_pos).length() < 0.5
            {
                self.window_open = !self.window_open;
            }
        }
    }

    fn size() -> Vec2 {
        vec2(500.0, 250.0)
    }

    fn slot(&mut self, ui: &mut megaui::Ui, s: SlotHandle, art: Option<Art>, p: Vec2) {
        use megaui::{widgets::Group, Drag, Vector2};

        let id = hash!(s);
        let (x, y) = p.into();
        let drag = Group::new(id, Vector2::new(50., 50.))
            .position(Vector2::new(x, y))
            .draggable(art.is_some())
            .hoverable(self.dragging && Some(s) != self.selected)
            .highlight(Some(s) == self.selected)
            .ui(ui, |ui| {
                ui.label(Vector2::new(5., 10.), art.as_ref().map(|a| a.name()).unwrap_or(""));
            });

        if ui.last_item_clicked() && art.is_some() {
            self.selected = Some(s);
        }

        match drag {
            Drag::Dropped(_, id) => {
                self.dragging = false;
                if let Some(&to) = id.and_then(|id| self.handle_bank.get(&id)) {
                    self.moves.push(ItemMove { from: s, to });
                }
            }
            Drag::Dragging(_, _) => {
                self.dragging = true;
            }
            Drag::No => {}
        }
    }

    fn ui(&mut self, bag: &mut Bag) {
        use megaui_macroquad::{
            draw_window,
            megaui::{
                widgets::{Group, Label},
                Vector2,
            },
            WindowParams,
        };

        if bag.new_item {
            self.icon_jumping = true;
            bag.new_item = false;
        }

        if !self.window_open {
            return;
        }
        self.icon_jumping = false;

        self.moves.drain_filter(|&mut ItemMove { from, to }| match [from, to] {
            [SlotHandle::Loose(i0), SlotHandle::Loose(i1)] => {
                let temp = bag.slots[i0].take();
                bag.slots[i0] = bag.slots[i1].take();
                bag.slots[i1] = temp;
                true
            }
            [SlotHandle::Trinket(i0), SlotHandle::Trinket(i1)] => {
                let temp = bag.trinkets[i0].take();
                bag.trinkets[i0] = bag.trinkets[i1].take();
                bag.trinkets[i1] = temp;
                true
            }
            [SlotHandle::Trinket(t), SlotHandle::Loose(l)]
            | [SlotHandle::Loose(l), SlotHandle::Trinket(t)] => {
                if let Some(Item::Trinket(loose_trinket)) = bag.slots[l] {
                    bag.slots[l] = bag.trinkets[t].map(|t| Item::Trinket(t));
                    bag.trinkets[t] = Some(loose_trinket);
                } else if bag.slots[l].is_none() {
                    bag.slots[l] = bag.trinkets[t].map(|t| Item::Trinket(t));
                    bag.trinkets[t] = None;
                }
                true
            }
            [SlotHandle::Weapon, SlotHandle::Loose(l)]
            | [SlotHandle::Loose(l), SlotHandle::Weapon] => {
                if !matches!(bag.slots[l], None | Some(Item::Weapon(_))) {
                    return true;
                }

                if let Some(wep) = bag.weapon.as_mut() {
                    if !wep.is_in() {
                        wep.get_in();
                        return false;
                    }

                    if let Some(old_wep) = bag.weapon.take().and_then(|w| w.take_item()) {
                        let new_wep = bag.slots[l].take();
                        bag.slots[l] = Some(Item::Weapon(old_wep));
                        if let Some(wep) = new_wep {
                            bag.take(wep);
                        }
                    }
                } else if let Some(wep) = bag.slots[l].take() {
                    bag.take(wep);
                }

                true
            }
            _ => true,
        });

        let size = Self::size();
        self.window_open = draw_window(
            hash!(),
            vec2(screen_width(), screen_height()) / 2.0 - size * vec2(0.6, 0.1),
            size,
            WindowParams { label: "Items - toggle with E".to_string(), ..Default::default() },
            |ui| {
                let pane = Vector2::new(size.x() * 0.3, size.y() - 25.0);
                Group::new(hash!(), pane).position(Vector2::new(5.0, 5.0)).ui(ui, |ui| {
                    ui.label(Vector2::new(80.0, 35.0), "<- Weapon");
                    self.slot(
                        ui,
                        SlotHandle::Weapon,
                        bag.weapon.as_ref().map(|w| w.item().art),
                        vec2(20.0, 20.0),
                    );

                    ui.label(Vector2::new(40.0, 85.0), "Trinkets");
                    for (i, trinket) in bag.trinkets.iter_mut().enumerate() {
                        self.slot(
                            ui,
                            SlotHandle::Trinket(i),
                            trinket.map(|t| Art::Trinket(t)),
                            vec2(
                                (i % 2) as f32 * 55.0 + 20.0,
                                (i as f32 / 2.0).floor() * 55.0 + 105.0,
                            ),
                        );
                    }
                });

                for (i, item) in bag.slots.iter_mut().enumerate() {
                    self.slot(
                        ui,
                        SlotHandle::Loose(i),
                        item.as_ref().map(|s| s.art()),
                        vec2(
                            (i % 3) as f32 * 55.0 + size.x() * 0.3 + 20.0,
                            (i as f32 / 3.0).floor() * 55.0 + 10.0,
                        ),
                    );
                }

                let trinket_space = bag.trinket_space().map(|i| SlotHandle::Trinket(i));

                let info = match self.selected {
                    Some(SlotHandle::Loose(i)) => {
                        bag.slots[i].as_ref().map(|b| (b.art(), b.description()))
                    }
                    Some(SlotHandle::Trinket(i)) => {
                        bag.trinkets[i].map(|t| (Art::Trinket(t), t.description()))
                    }
                    Some(SlotHandle::Weapon) => {
                        bag.weapon.as_ref().map(|b| (b.item().art, b.item().description()))
                    }
                    _ => None,
                };

                if let Some((art, desc)) = info {
                    let name = art.name();

                    Group::new(hash!(), pane).position(Vector2::new(345.0, 5.0)).ui(ui, |ui| {
                        Group::new(hash!(), Vector2::new(100.0, 100.0))
                            .position(Vector2::new((pane.x - 100.0) / 2.0, 10.0))
                            .ui(ui, |ui| ui.label(None, name));

                        ui.label(Vector2::new(5.0, 115.0), name);
                        Label::new(desc).position(Vector2::new(5.0, 135.0)).multiline(14.0).ui(ui);

                        let bottom = Vector2::new(5.0, 202.0);
                        if let Some(SlotHandle::Loose(i)) = self.selected {
                            match bag.slots[i].as_ref() {
                                Some(&Item::Trinket(_)) => {
                                    if let Some(space) = trinket_space {
                                        if ui.button(bottom, "Equip") {
                                            self.moves.push(ItemMove {
                                                from: SlotHandle::Loose(i),
                                                to: space,
                                            });
                                            self.selected = Some(space);
                                        }
                                    } else {
                                        ui.label(bottom, "Clear space to equip");
                                    }
                                }
                                Some(&Item::Consumable(consumable)) => {
                                    if ui.button(bottom, "Consume") {
                                        bag.slots[i].take();
                                        self.stomach.push(consumable);
                                    }
                                }
                                _ => {}
                            }
                        }
                    });
                }
            },
        );
    }
}

struct WantsIn {
    bag_ent: hecs::Entity,
    wep_ent: hecs::Entity,
}

struct KeeperOfBags {
    wants_out: Vec<(hecs::Entity, WeaponItem)>,
    wants_in: Vec<WantsIn>,
    ownerless: Vec<hecs::Entity>,
}
impl KeeperOfBags {
    fn new() -> Self {
        Self {
            wants_out: Vec::with_capacity(100),
            wants_in: Vec::with_capacity(100),
            ownerless: Vec::with_capacity(100),
        }
    }

    fn keep(&mut self, ecs: &mut hecs::World) {
        let Self { wants_out, wants_in, ownerless } = self;
        wants_out.extend(
            ecs.query::<&mut Bag>()
                .iter()
                .filter(|(_, b)| b.weapon.as_ref().map(|b| b.wants_out()).unwrap_or(false))
                .filter_map(|(_, b)| {
                    let item = b.weapon.take()?.take_item()?;
                    let ent = ecs.reserve_entity();
                    b.weapon = Some(BagWeapon::Out { ent, item: item.clone(), wants_in: false });
                    Some((ent, item))
                }),
        );
        for (ent, weapon) in wants_out.drain(..) {
            or_err!(ecs.insert(ent, weapon));
        }

        wants_in.extend(
            ecs.query::<&mut Bag>()
                .iter()
                .filter(|(_, b)| b.weapon.as_ref().map(|b| b.wants_in()).unwrap_or(false))
                .filter_map(|(e, b)| {
                    Some(WantsIn { bag_ent: e, wep_ent: b.weapon.as_ref().and_then(|w| w.ent())? })
                }),
        );
        for WantsIn { bag_ent, wep_ent } in wants_in.drain(..) {
            let item = ecs.remove(wep_ent);
            if let (Ok(item), Ok(mut bag)) = (item, ecs.get_mut::<Bag>(bag_ent)) {
                bag.weapon = Some(BagWeapon::In { item, wants_out: false });
            }
        }

        ownerless.extend(
            ecs.query::<&WeaponState>()
                .iter()
                .filter(|(_, w)| !ecs.contains(w.owner))
                .map(|(e, _)| e),
        );
        for w in ownerless.drain(..) {
            or_err!(ecs.despawn(w));
        }
    }
}

#[derive(Debug, Clone)]
struct WeaponState {
    last_rot: Rot,
    owner: hecs::Entity,
    attack: Attack,
    ents_hit: smallvec::SmallVec<[hecs::Entity; 5]>,
}
impl WeaponState {
    fn new(owner: hecs::Entity) -> Self {
        Self {
            owner,
            last_rot: Default::default(),
            attack: Default::default(),
            ents_hit: Default::default(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct WeaponHit {
    hit: hecs::Entity,
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
    attack_kind: AttackKind,
    start_tick: u32,
    end_tick: u32,
    toward: Vec2,
}

#[derive(Debug, Clone, Copy)]
enum AttackKind {
    Swipe,
    Stab,
}
impl AttackKind {
    fn duration(self) -> u32 {
        match self {
            AttackKind::Swipe => 50,
            AttackKind::Stab => 30,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WeaponInput {
    start_attacking: Option<AttackKind>,
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

    /// Returns the direction the weapon is swinging toward if it is swinging.
    fn swing_dir(&self) -> Option<Vec2> {
        if let Attack::Swing(SwingState { toward, .. }) = self.wep.attack {
            Some(toward)
        } else {
            None
        }
    }

    /// The exact middle of the swing has been reached
    fn mid_swing(&self, tick: u32) -> bool {
        if let Attack::Swing(SwingState { end_tick, .. }) = self.wep.attack {
            end_tick - tick == 25
        } else {
            false
        }
    }

    fn attack_hits(&mut self) -> smallvec::SmallVec<[WeaponHit; 5]> {
        match self.wep.attack {
            Attack::Swing(s) if s.doing_damage => self
                .contacts
                .0
                .iter()
                .filter(|h| matches!(h.circle_kinds, [CircleKind::Hurt, CircleKind::Hit]))
                .map(|h| WeaponHit { hit: h.hit })
                .filter(|wh| {
                    if self.wep.ents_hit.contains(&wh.hit) {
                        false
                    } else {
                        self.wep.ents_hit.push(wh.hit);
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
        self.wep.attack = match (start_attacking, self.wep.attack) {
            (Some(attack_kind), Ready) => {
                self.wep.ents_hit.clear();
                Swing(SwingState {
                    start_tick: tick,
                    end_tick: tick + attack_kind.duration(),
                    attack_kind,
                    toward: self.from_center(wielder_pos, target),
                    doing_damage: false,
                })
            }
            (_, Swing(SwingState { end_tick, .. })) if end_tick < tick => {
                self.wep.ents_hit.clear();
                Cooldown { end_tick: tick + 10 }
            }
            (_, Cooldown { end_tick, .. }) if end_tick < tick => Ready,
            (_, other) => other,
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
        SwingState { attack_kind, start_tick, end_tick, toward, .. }: SwingState,
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
        let frames = match attack_kind {
            AttackKind::Swipe => [
                (2.50, Some(rot - swing * 1.0), Some(hand_pos) , false), // ready   
                (2.65, Some(rot - swing * 2.0), None           , false), // back up 
                (1.00, Some(rot + swing * 2.0), None           , true ), // swipe   
                (2.85, Some(rot + swing * 3.0), None           , false), // recovery
                (2.50, Some(start_rot.0      ), Some(start_pos), false), // return  
            ],
            AttackKind::Stab => [
                (3.00, Some(rot)        , Some(hand_pos               ), false), // ready   
                (2.00, None             , Some(hand_pos - toward * 0.2), false), // back up 
                (1.00, None             , Some(hand_pos + toward * 1.2), true ), // stab   
                (2.50, None             , Some(hand_pos               ), false), // recovery
                (2.50, Some(start_rot.0), Some(start_pos              ), false), // return  
            ]
        };

        let (st, et, t) = (start_tick as f32, end_tick as f32, tick as f32);
        let dt = et - st;
        let total = frames.iter().map(|&(t, ..)| t).sum::<f32>();

        let mut last_tick = st;
        let mut last_rot = start_rot.vec2();
        let mut last_pos = start_pos;
        let mut do_damage = false;
        for &(duration, angle_rot, pos, damage) in &frames {
            let tick = last_tick + dt * (duration / total);
            let prog = math::inv_lerp(last_tick, tick, t).min(1.0).max(0.0);

            if prog > 0.0 {
                if prog < 1.0 {
                    do_damage = do_damage || damage;
                }

                if let Some(angle_rot) = angle_rot {
                    let rot = Rot(angle_rot).vec2();
                    self.rot.set_vec2(math::slerp(last_rot, rot, prog));
                    last_rot = rot;
                }
                if let Some(p) = pos {
                    *self.pos = last_pos.lerp(p, prog);
                    last_pos = p;
                }
            }

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
    hp: &'a mut Health,
}
impl Hero<'_> {
    fn movement(&mut self, mods: HeroMod, hero_swinging: bool) {
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
                * if hero_swinging { mods.swinging_movement } else { 1.0 }
                * if self.bag.weapon.is_none()
                    || move_vec.x() == 0.0
                    || move_vec.x().signum() == self.dir.signum()
                {
                    0.0075 * mods.forward_movement
                } else {
                    0.0045 * mods.backward_movement
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

    fn consume(&mut self, consumable: Consumable) {
        match consumable {
            Consumable::HealthPotion => self.hp.0 = (self.hp.0 + 8).min(self.hp.1),
        }
    }
}

const GOLDEN_RATIO: f32 = 1.618034;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Art {
    Hero,
    Consumable(Consumable),
    Trinket(Trinket),
    Fireball,
    Compass,
    Scarecrow,
    Arrow,
    Chest,
    Lockbox,
    TripletuftedTerrorworm,
    VioletVagabond,
    Tree,
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

    fn name(self) -> &'static str {
        match self {
            Art::Hero => "Hero",
            Art::Consumable(c) => c.name(),
            Art::Trinket(t) => t.name(),
            Art::Fireball => "Fireball",
            Art::Scarecrow => "Scarecrow",
            Art::Arrow => "Arrow",
            Art::Chest => "Chest",
            Art::Lockbox => "Lockbox",
            Art::Compass => "Compass",
            Art::TripletuftedTerrorworm => "Tripletufted Terrorworm",
            Art::VioletVagabond => "Violet Vagabond",
            Art::Tree => "Tree",
            Art::Npc => "Npc",
            Art::Sword => "Sword",
            Art::Post => "Post",
        }
    }

    fn bounding(self) -> f32 {
        match self {
            Art::Arrow | Art::Chest | Art::Npc | Art::Sword => GOLDEN_RATIO / 2.0,
            Art::Hero => 0.5,
            Art::Lockbox => 0.5,
            Art::Compass => 0.5,
            Art::TripletuftedTerrorworm | Art::VioletVagabond | Art::Scarecrow => 0.4,
            Art::Fireball => 0.5,
            Art::Tree => 2.0,
            Art::Post => 1.05,
            Art::Trinket(_) | Art::Consumable(_) => 0.0,
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
struct Health(u32, u32);
impl Health {
    fn full(hp: u32) -> Self {
        Self(hp, hp)
    }

    fn ratio(self) -> f32 {
        self.0 as f32 / self.1 as f32
    }
}

#[derive(Copy, Clone, Debug)]
struct HealthBar;

macro_rules! trinkets {
    ( $($name:ident : {
        name: $display_name:literal,
        mods: $mods:expr,
        desc: $desc:expr,
    })* ) => {
        #[derive(Copy, Clone, Debug, PartialEq)]
        enum Trinket { $( $name, )* }
        impl Trinket {
            fn mods(self) -> HeroMod {
                match self {
                    $( Trinket::$name => $mods, )*
                }
            }

            fn name(self) -> &'static str {
                match self {
                    $( Trinket::$name => $display_name, )*
                }
            }

            fn description(self) -> &'static str {
                match self {
                    $( Trinket::$name => $desc, )*
                }
            }
        }
    }
}

trinkets! {
    StoneRing: {
        name: "Stone Ring",
        mods: HeroMod { swinging_movement: -0.15, crit_chance: 0.05, ..HeroMod::empty() },
        desc: concat!(
            "Move 15% slower \n",
            "while swinging, \n",
            "Land critical hits \n",
            "5% more often. \n",
        ),
    }
    VolcanicShard: {
        name: "Volcanic Shard",
        mods: HeroMod { fireball_chance: 0.05, ..HeroMod::empty() },
        desc: concat!(
            "Shard of black glass. \n",
            "Grants each attack \n",
            "a 5% chance of \n",
            "launching a fireball. \n",
        ),
    }
    LoftyInsoles: {
        name: "Lofty Insoles",
        mods: HeroMod { forward_movement: 0.15, backward_movement: -0.15, ..HeroMod::empty() },
        desc: concat!(
            "Seems to caress your toes. \n",
            "Move forward  15% faster, \n",
            "Move backward 15% slower. \n",
        ),
    }
}

#[derive(Clone, Copy, Debug)]
struct HeroMod {
    swinging_movement: f32,
    forward_movement: f32,
    backward_movement: f32,
    crit_chance: f32,
    fireball_chance: f32,
}
impl HeroMod {
    fn empty() -> Self {
        Self {
            swinging_movement: 0.0,
            forward_movement: 0.0,
            backward_movement: 0.0,
            crit_chance: 0.0,
            fireball_chance: 0.0,
        }
    }
}
impl std::ops::Add<HeroMod> for HeroMod {
    type Output = Self;

    fn add(self, othr: HeroMod) -> Self::Output {
        Self {
            swinging_movement: self.swinging_movement + othr.swinging_movement,
            forward_movement: self.forward_movement + othr.forward_movement,
            backward_movement: self.backward_movement + othr.backward_movement,
            crit_chance: self.crit_chance + othr.crit_chance,
            fireball_chance: self.fireball_chance + othr.fireball_chance,
        }
    }
}

impl Default for HeroMod {
    fn default() -> Self {
        HeroMod {
            swinging_movement: 1.0,
            forward_movement: 1.0,
            backward_movement: 1.0,
            crit_chance: 0.01,
            fireball_chance: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Consumable {
    HealthPotion,
}
impl Consumable {
    fn name(self) -> &'static str {
        match self {
            Consumable::HealthPotion => "Health Potion",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Consumable::HealthPotion => concat!(
                "Theoretically \n",
                "restores 8 \n",
                "hitpoints. Not \n",
                "clinically tested. \n",
            ),
        }
    }
}

#[derive(Clone, Debug)]
enum Item {
    Weapon(WeaponItem),
    Trinket(Trinket),
    Consumable(Consumable),
}
impl Item {
    fn art(&self) -> Art {
        match self {
            &Item::Trinket(t) => Art::Trinket(t),
            &Item::Consumable(t) => Art::Consumable(t),
            &Item::Weapon(WeaponItem { art, .. }) => art,
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Item::Trinket(t) => t.description(),
            Item::Consumable(c) => c.description(),
            Item::Weapon(w) => w.description(),
        }
    }
}

impl Item {
    fn sword(owner: hecs::Entity) -> Self {
        Item::Weapon(WeaponItem {
            pos: Vec2::zero(),
            art: Art::Sword,
            rot: Rot(0.0),
            phys: Phys::with(&[
                Circle::hurt(0.2, vec2(0.0, 1.35)),
                Circle::hurt(0.185, vec2(0.0, 1.1)),
                Circle::hurt(0.15, vec2(0.0, 0.85)),
                Circle::hurt(0.125, vec2(0.0, 0.65)),
            ]),
            wep: WeaponState::new(owner),
            contacts: Contacts::default(),
        })
    }
}

#[derive(hecs::Bundle, Clone, Debug)]
struct WeaponItem {
    pos: Vec2,
    art: Art,
    rot: Rot,
    phys: Phys,
    wep: WeaponState,
    contacts: Contacts,
}
impl WeaponItem {
    fn description(&self) -> &'static str {
        concat!(
            "A dangerous weapon. \n",
            "Try not to poke \n",
            "your eyes out, like \n",
            "the last adventurer. \n",
        )
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

#[derive(Copy, Clone)]
struct Fireball {
    damage: u32,
    knockback: f32,
    crit_chance: f32,
}

struct FireballHit {
    fireball_ent: hecs::Entity,
    fireball: Fireball,
    hit: hecs::Entity,
    pos: Vec2,
}
struct Fireballs {
    dead: Vec<FireballHit>,
}
impl Fireballs {
    fn new() -> Self {
        Self { dead: Vec::with_capacity(100) }
    }

    fn update(&mut self, game: &mut Game, quests: &mut Quests) {
        let Game { ecs, .. } = game;

        system!(ecs, ent,
            Contacts(hits) = &mut _
            &pos           = &Vec2
            &fireball      = &Fireball
        {
            if let Some(Hit { hit, .. }) = hits.pop() {
                self.dead.push(FireballHit { fireball_ent: ent, fireball, hit, pos });
            }
        });

        for FireballHit { fireball_ent, fireball, hit, pos } in self.dead.drain(..) {
            let (dead, _) = game.hit(HitInput {
                hitter_pos: pos,
                hit: WeaponHit { hit },
                damage: fireball.damage,
                knockback: fireball.knockback,
                crit_chance: fireball.crit_chance,
            });
            if dead {
                quests.update(game);
            }

            or_err!(game.ecs.despawn(fireball_ent));
        }
    }
}

fn update_hero(hero: hecs::Entity, game: &mut Game, quests: &mut Quests, bag_ui: &mut BagUi) {
    let &Game { tick, mouse_pos, .. } = &*game;
    let Game { ecs, hero_swinging, .. } = game;

    let (hero_pos, hero_vel, hero_dir, hero_wep, hero_mods) = match ecs.query_one_mut::<Hero>(hero)
    {
        Ok(mut hero) => {
            let mods = hero.bag.mods();

            hero.movement(mods, *hero_swinging);

            for consumable in bag_ui.stomach.drain(..) {
                hero.consume(consumable);
            }

            (*hero.pos, hero.vel.0, *hero.dir, hero.bag.weapon.as_mut().and_then(|w| w.out()), mods)
        }
        Err(_) => return,
    };
    game.hero_pos = hero_pos;

    let mut swing_dir = None;
    let mut mid_swing = false;
    let hero_attacks = hero_wep
        .and_then(|e| ecs.query_one_mut::<Weapon>(e).ok())
        .map(|mut wep| {
            wep.tick(WeaponInput {
                start_attacking: Some(AttackKind::Swipe)
                    .filter(|_| !mouse_over_ui() && is_mouse_button_down(MouseButton::Left)),
                target: mouse_pos,
                wielder_pos: hero_pos,
                wielder_vel: hero_vel,
                wielder_dir: hero_dir,
                tick,
            });
            mid_swing = wep.mid_swing(tick);
            swing_dir = wep.swing_dir();
            *hero_swinging = wep.swing_dir().is_some();
            wep.attack_hits()
        })
        .unwrap_or_default();

    let mut knockback = hero_attacks.iter().fold(Vec2::zero(), |acc, &hit| {
        let (dead, vel) = game.hit(HitInput {
            hitter_pos: hero_pos,
            hit,
            damage: 1,
            knockback: 0.2,
            crit_chance: hero_mods.crit_chance,
        });
        if dead {
            quests.update(game);
        }
        acc + vel
    });

    if let Some(dir) = swing_dir
        .filter(|_| mid_swing)
        .filter(|_| rand::gen_range(0.0, 1.0) < hero_mods.fireball_chance)
    {
        knockback -= dir * 0.35;
        game.ecs.spawn((
            Art::Fireball,
            Fireball { damage: 3, knockback: 0.2, crit_chance: hero_mods.crit_chance },
            Phys::new()
                .insert(Circle::hurt(0.4, Vec2::zero()))
                .insert(Circle::push(0.1, Vec2::zero())),
            Velocity(dir / 6.0),
            MinVelocity(dir / 15.0),
            hero_pos + dir * 2.0 + vec2(0.0, 0.5),
            Rot(Rot::from_vec2(dir).0 + PI),
            Contacts::default(),
        ));
    }

    if let Ok(mut v) = game.ecs.get_mut::<Velocity>(hero) {
        v.knockback(knockback, 0.125);
    }
}

#[derive(Copy, Clone, Debug)]
enum FighterBehavior {
    SwingAlways,
    LowHealthStab,
}
impl FighterBehavior {
    fn attack_kind(self, hp: Health) -> AttackKind {
        use {AttackKind::*, FighterBehavior::*};
        match self {
            LowHealthStab if hp.ratio() < 0.5 => Stab,
            _ => Swipe,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Fighter {
    behavior: FighterBehavior,
    aggroed: bool,
    chase_speed: f32,
    charge_speed: f32,
    attack_interval: u32,
}
impl Default for Fighter {
    fn default() -> Self {
        Self {
            behavior: FighterBehavior::SwingAlways,
            aggroed: false,
            chase_speed: 0.0065,
            charge_speed: 0.0085,
            attack_interval: 120,
        }
    }
}

struct Waffle {
    enemies: Vec<(hecs::Entity, Vec2, Velocity, Fighter, Health)>,
    last_attack: u32,
    attack_ends: u32,
    attack_kind: Option<AttackKind>,
    attacker: Option<hecs::Entity>,
    occupied_slots_last_frame: usize,
}
impl Waffle {
    fn new() -> Self {
        Self {
            enemies: Vec::with_capacity(100),
            last_attack: 0,
            attack_ends: 0,
            attacker: None,
            attack_kind: None,
            occupied_slots_last_frame: 0,
        }
    }

    fn update(&mut self, game: &mut Game) {
        let &mut Game { hero_pos, tick, .. } = game;
        let Self {
            enemies,
            attack_ends,
            last_attack,
            attack_kind,
            attacker,
            occupied_slots_last_frame,
        } = self;

        let ecs = &mut game.ecs;
        system!(ecs, _,
            fighter = &mut Fighter
            &pos    = &Vec2
        {
            let dist = (hero_pos - pos).length();
            fighter.aggroed = if dist < 5.0 {
                true
            } else if dist > 15.0 {
                false
            } else {
                fighter.aggroed
            };
        });

        enemies.extend(
            ecs.query::<(&_, &_, &Fighter, &_)>()
                .iter()
                .filter(|(_, (_, _, f, ..))| f.aggroed)
                .map(|(e, (p, v, f, h))| (e, *p, *v, *f, *h)),
        );
        enemies.sort_by(|a, b| {
            if Some(a.0) == *attacker {
                std::cmp::Ordering::Less
            } else {
                float_cmp(a, b, |&(_, p, ..)| (hero_pos - p).length_squared())
            }
        });

        const SLOT_COUNT: usize = 7;
        let mut slots = [false; SLOT_COUNT];
        for (e, pos, vel, f, hp) in enemies.drain(..) {
            let slot = circle_points(SLOT_COUNT)
                .map(|v| (Rot(0.1).apply(v) * 2.65 + hero_pos) - pos)
                .enumerate()
                .filter(|&(i, _)| !slots[i])
                .min_by(|a, b| float_cmp(a, b, |d| d.1.length_squared()));

            if let Some((i, delta)) = slot {
                slots[i] = true;

                if *attack_ends < tick && (Some(e) != *attacker || *occupied_slots_last_frame == 1)
                {
                    *attacker = Some(e);
                    *last_attack = tick;
                    *attack_ends = tick + f.attack_interval;
                    *attack_kind = Some(f.behavior.attack_kind(hp));
                }

                if let Ok(Velocity(vel)) = game.ecs.get_mut(e).as_deref_mut() {
                    let (d, speed) = if Some(e) == *attacker && *last_attack + 40 > tick {
                        let goal = hero_pos + (pos - hero_pos).normalize() * 2.0;
                        ((goal - pos), f.charge_speed)
                    } else {
                        (delta, f.chase_speed)
                    };

                    *vel += d.min(d.normalize()) * speed;
                }
            }

            if let Ok(Some(wep_ent)) =
                game.ecs.get_mut::<Bag>(e).map(|mut b| b.weapon.as_mut().and_then(|w| w.out()))
            {
                let attacks = game
                    .ecs
                    .query_one_mut::<Weapon>(wep_ent)
                    .ok()
                    .map(|mut wep| {
                        wep.tick(WeaponInput {
                            start_attacking: attack_kind
                                .filter(|_| *attacker == Some(e) && *last_attack == tick),
                            target: hero_pos,
                            wielder_pos: pos,
                            wielder_vel: vel.0,
                            wielder_dir: (hero_pos - pos).x().into(),
                            tick,
                        });
                        wep.attack_hits()
                    })
                    .unwrap_or_default();

                if let (Some(attack_kind), true) = (*attack_kind, *attacker == Some(e)) {
                    let (damage, attacker_knockback, knockback) = match attack_kind {
                        AttackKind::Swipe => (1, 0.125, 0.2),
                        AttackKind::Stab => (3, 0.085, 0.28),
                    };
                    let delta = attacks.iter().fold(Vec2::zero(), |acc, &hit| {
                        let (_, vel) = game.hit(HitInput {
                            hitter_pos: pos,
                            hit,
                            damage,
                            knockback,
                            crit_chance: 0.0,
                        });
                        acc + vel
                    });
                    if let Ok(mut v) = game.ecs.get_mut::<Velocity>(e) {
                        v.knockback(delta, attacker_knockback);
                    }
                }
            }
        }
        *occupied_slots_last_frame = slots.iter().filter(|&&s| s).count();
    }
}

fn paragraph(ui: &mut megaui::Ui, para: &'static str) {
    use megaui::widgets::Label;
    Label::new(para).multiline(14.0).ui(ui);
}

#[derive(Copy, Clone)]
struct RewardChoice {
    id: u64,
    choices: &'static [(Item, usize, &'static str, Option<&'static str>)],
    selected: Option<Art>,
    rewarded: bool,
}
impl RewardChoice {
    fn new(id: u64, choices: &'static [(Item, usize, &'static str, Option<&'static str>)]) -> Self {
        Self { id, choices, selected: None, rewarded: false }
    }

    fn ui(&mut self, ui: &mut megaui::Ui) -> smallvec::SmallVec<[Item; 5]> {
        use megaui::{
            widgets::{Group, Label},
            Vector2,
        };
        let &mut Self { id, ref mut selected, choices, ref mut rewarded } = self;
        let mut selected_text = None;
        let mut accepted = false;

        paragraph(ui, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        ui.group(hash!(id, "choicebox"), Vector2::new(255.0, 200.0), |ui| {
            for (i, &(ref item, _, text, longtext)) in choices.iter().enumerate() {
                let art = item.art();

                Group::new(hash!(id, "choice", text), Vector2::new(60.0, 80.0))
                    .position(Vector2::new(47.0 + i as f32 / choices.len() as f32 * 200.0, 10.0))
                    .highlight(*selected == Some(art))
                    .ui(ui, |ui| {
                        Group::new(hash!(id, "choiceinner", text), Vector2::new(50.0, 50.0))
                            .position(Vector2::new(5.0, 5.0))
                            .ui(ui, |ui| ui.label(None, art.name()));

                        if !*rewarded && ui.button(Vector2::new(12.0, 58.0), "Take") {
                            *selected = Some(art);
                        }
                    });

                if *selected == Some(art) {
                    selected_text = Some((text, longtext.unwrap_or(item.description())));
                }
            }

            if let Some((t, longt)) = selected_text {
                ui.label(Vector2::new(5.0, 100.0), t);

                Label::new(longt).position(Vector2::new(5.0, 120.0)).multiline(14.0).ui(ui);

                if !*rewarded {
                    accepted = ui.button(Vector2::new(5.0, 177.0), &format!("Confirm {}", t));
                } else {
                    ui.label(Vector2::new(5.0, 177.0), &format!("{} Accepted", t));
                }
            }
        });

        selected
            .filter(|_| {
                if accepted && !*rewarded {
                    *rewarded = true;
                    true
                } else {
                    false
                }
            })
            .and_then(|art| {
                self.choices
                    .iter()
                    .find(|(i, ..)| i.art() == art)
                    .map(|(i, c, ..)| (0..*c).map(|_| i.clone()).collect())
            })
            .unwrap_or_default()
    }
}

struct Npc {
    name: &'static str,
    talks: Vec<Box<dyn FnMut(&mut megaui::Ui, &mut Game) + Send + Sync>>,
    within_range: bool,
}
impl Npc {
    fn new(name: &'static str) -> Self {
        Self { name, talks: vec![], within_range: false }
    }

    fn with_paragraph(mut self, para: &'static str) -> Self {
        self.ui(move |ui, _| paragraph(ui, para));
        self
    }

    fn ui(&mut self, f: impl FnMut(&mut megaui::Ui, &mut Game) + Send + Sync + 'static) {
        self.talks.push(Box::new(f));
    }
}

struct NpcUi {
    talks: std::collections::HashMap<
        &'static str,
        (bool, Vec<Box<dyn FnMut(&mut megaui::Ui, &mut Game) + Send + Sync>>),
    >,
}

impl NpcUi {
    fn new() -> Self {
        Self { talks: std::collections::HashMap::with_capacity(100) }
    }

    fn ui(&mut self, game: &mut Game) {
        use megaui_macroquad::{draw_window, WindowParams};
        let ecs = &mut game.ecs;
        let hero_pos = game.hero_pos;

        for (_, (npc, &pos)) in &mut ecs.query::<(&mut Npc, &Vec2)>() {
            let (active, talks) =
                self.talks.entry(npc.name).or_insert((false, Vec::with_capacity(10)));

            let within_range = (pos - hero_pos).length() < 1.3;
            if within_range != npc.within_range {
                *active = within_range;
            }
            npc.within_range = within_range;

            talks.append(&mut npc.talks);
        }
        let i = self.talks.values().filter(|(a, _)| *a).count();

        for (name, (active, talks)) in self.talks.iter_mut().filter(|(_, (a, _))| *a) {
            let size = vec2(282.0, 400.0);
            *active = draw_window(
                hash!(name),
                vec2(screen_width(), screen_height()) / 2.0 - size * (0.4 + 0.05 * i as f32),
                size,
                WindowParams { label: name.to_string(), close_button: true, ..Default::default() },
                |ui| {
                    if let Some(top) = talks.last_mut() {
                        open_tree(ui, hash!(name, "cur"), "Greetings!", |ui| (top)(ui, game));
                        ui.separator();
                    }

                    if talks.len() < 2 {
                        return;
                    }

                    ui.tree_node(hash!(name, "old"), "Older Monologues", |ui| {
                        for talk in talks.iter_mut().rev().skip(1) {
                            (talk)(ui, game);
                            ui.separator();
                        }
                    });
                },
            );
        }
    }
}

#[macroquad::main("rpg")]
async fn main() {
    let mut ecs = hecs::World::new();
    let mut drawer = Drawer::new();
    let mut waffle = Waffle::new();
    let mut fireballs = Fireballs::new();
    let mut physics = Physics::new();
    let mut bag_ui = BagUi::new(&mut ecs);
    let mut npc_ui = NpcUi::new();
    let mut quests = Quests::new(&mut ecs);
    let mut keeper_of_bags = KeeperOfBags::new();
    let mut fps = Fps::new();

    let npc =
        |npc_pos, npc| (npc_pos, npc, Phys::new().wings(0.3, 0.21, CircleKind::Push), Art::Npc);

    let hero = ecs.spawn((
        vec2(-0.4, -3.4),
        Velocity::default(),
        Phys::new().wings(0.3, 0.21, CircleKind::Push).hitbox(0.5),
        Art::Hero,
        Direction::Left,
        Health::full(10),
        Bag::default(),
    ));
    ecs.spawn((
        vec2(0.0, 0.0),
        Art::Chest,
        Phys::new().wings(0.4, 0.4, CircleKind::Push).wings(0.4, 0.4, CircleKind::Hit),
        Velocity::default(),
        Health::full(5),
    ));

    type Post = (Vec2, Art, Phys);
    #[derive(Clone, Copy)]
    struct Pen {
        gate: [(hecs::Entity, Post); 2],
        pos: Vec2,
        radius: f32,
    }
    impl Pen {
        fn new(ecs: &mut hecs::World, radius: f32, pos: Vec2, gate: usize) -> Self {
            let mut pen =
                Self { gate: unsafe { [std::mem::zeroed(), std::mem::zeroed()] }, radius, pos };

            for (i, post_pos) in circle_points((PI * radius / 0.4).ceil() as usize).enumerate() {
                let post = pen.post(post_pos);
                let e = ecs.spawn(post.clone());

                let gate_i = i - gate;
                if gate_i <= 1 {
                    pen.gate[gate_i] = (e, post);
                }
            }

            pen
        }

        fn post(&self, post_pos: Vec2) -> Post {
            (post_pos * self.radius + self.pos, Art::Post, Phys::new().pushbox(0.4))
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

    fn tree(ecs: &mut hecs::World, p: Vec2) {
        ecs.spawn((p, Art::Tree, Phys::new().pushbox(0.4)));
    }
    for p in circle_points(5) {
        tree(&mut ecs, p * 6.0 + vec2(-21.5, -20.0));
    }
    for p in circle_points(8) {
        tree(&mut ecs, p * 18.0 + vec2(-21.5, -20.0));
    }
    for p in circle_points(6) {
        tree(&mut ecs, p * 11.0 + vec2(5.0, 5.0));
    }
    for p in circle_points(8) {
        tree(&mut ecs, p * 13.5 + vec2(-40.0, 10.0));
    }

    let vv = ecs.reserve_entity();
    or_err!(ecs.insert(
        vv,
        (
            vec2(-21.5, -20.0),
            Health::full(7),
            HealthBar,
            Art::VioletVagabond,
            Fighter {
                behavior: FighterBehavior::LowHealthStab,
                chase_speed: 0.0085,
                charge_speed: 0.0125,
                attack_interval: 75,
                ..Default::default()
            },
            Phys::new().pushfoot_bighit(),
            Velocity::default(),
            Bag::holding(Item::sword(vv)),
        )
    ));

    let lair_pen = Pen::new(&mut ecs, 8.0, vec2(-40.0, 10.0), 52);
    lair_pen.gate_open(&mut ecs);

    let mut tasks = vec![];

    let sword = ecs.spawn((vec2(-2.5, -0.4), Art::Sword, Rot(FRAC_PI_2 + 0.1), ZOffset(0.42)));

    let npc1 = ecs.spawn(npc(
        vec2(-4.0, 0.0),
        Npc::new("NPC 1").with_paragraph(concat!(
            "It's dangerous to go alone! \n",
            "Allow me to give you, a complete \n",
            "stranger, a dangerous weapon \n",
            "because I just happen to believe \n",
            "that you may be capable and willing \n",
            "to spare us from the horrible fate \n",
            "that will surely befall us regardless \n",
            "of whether or not you spend three hours \n",
            "immersed in a fishing minigame.",
        )),
    ));

    tasks.push(Task {
        label: "Navigate to the Sword",
        req: Box::new(move |g| g.hero_dist(g.pos(sword) - vec2(0.8, 0.7)) < 1.3),
        guide: Box::new(move |g, gi| gi.push((Art::Arrow, g.pos(sword) - vec2(0.8, 0.7)))),
        on_finish: Box::new(move |g| {
            or_err!(g.ecs.despawn(sword));
            or_err!(g.give_item(hero, Item::sword(hero)));
            if let Ok(mut npc) = g.ecs.get_mut::<Npc>(npc1) {
                npc.ui(|ui, _| {
                    paragraph(
                        ui,
                        concat!(
                            "I'm going to have to ask you to \n",
                            "ruthlessly destroy a scarecrow I \n",
                            "painstakingly made from scratch \n",
                            "over the course of several days \n",
                            "because this game takes place \n",
                            "before modern manufacturing practices, \n",
                            "but it's fine because it's not \n",
                            "like I have anything else to do other \n",
                            "than stand here pretending to work on \n",
                            "another one!",
                        ),
                    )
                });
            }
        }),
        ..Default::default()
    });

    let scarecrow = ecs.spawn((
        vec2(3.4, -2.4),
        Phys::new().pushfoot_bighit(),
        Health::full(5),
        HealthBar,
        Art::Scarecrow,
    ));

    tasks.push(Task {
        label: "Destroy Scarecrow",
        req: Box::new(move |g| g.dead(scarecrow)),
        guide: Box::new(move |g, gi| gi.push((Art::Sword, g.pos(scarecrow)))),
        on_finish: Box::new({
            use {Consumable::*, Trinket::*};

            #[rustfmt::skip]
            let mut reward = RewardChoice::new(hash!(), &[
                (Item::Consumable(HealthPotion), 5, "x5 Health Potions", Some(concat!(
                    "40 hitpoints worth of regeneration. \n",
                    "You have 10 hitpoints currently. \n",
                ))),
                (Item::Trinket(StoneRing), 1, "Stone Ring", None),
            ]);

            move |g| {
                if let Ok(mut npc) = g.ecs.get_mut::<Npc>(npc1) {
                    npc.ui(move |ui, g| {
                        paragraph(
                            ui,
                            concat!(
                                "As you have proven yourself \n",
                                "capable of obliterating inanimate \n",
                                "objects which are rarely fearsome \n",
                                "enough even to repel birds \n",
                                "characterized by their extreme \n",
                                "cowardice, any doubts I may have \n",
                                "had as to your ability to save us \n",
                                "from inescapable doom are certainly \n",
                                "assuaged. Allow me to award you with \n",
                                "some token of my appreciation. \n",
                            ),
                        );

                        for item in reward.ui(ui).into_iter() {
                            or_err!(g.give_item(hero, item));
                        }

                        paragraph(
                            ui,
                            concat!(
                                "I urge you to talk to NPC 2, who will \n",
                                "open the gate for you. Once inside \n",
                                "the pen, attempt to slaughter the \n",
                                "innocent looking creatures inside. \n",
                                "Show no remorse, adventurer. Soon \n",
                                "our plight will be known to you.",
                            ),
                        )
                    });
                }
            }
        }),
        ..Default::default()
    });

    let npc2 = ecs.spawn(npc(
        vec2(4.0, 11.0),
        Npc::new("NPC 2").with_paragraph(concat!(
            "Greetings! \n",
            "I can open this gate for you, \n",
            "but you don't look ready. \n",
        )),
    ));
    let npc2_guide =
        move |g: &Game, gi: &mut Vec<_>| gi.push((Art::Arrow, g.pos(npc2) + vec2(0.0, 0.9)));

    let pen = Pen::new(&mut ecs, 4.5, vec2(5.0, 5.0), 10);
    let (pen_pos, pen_radius) = (pen.pos, pen.radius);
    #[derive(Copy, Clone)]
    struct Terrorworms([hecs::Entity; 6]);
    impl Terrorworms {
        fn new(ecs: &mut hecs::World, pen: Pen) -> Self {
            let r = pen.radius * 0.4;
            let mut circle = circle_points(6);
            Terrorworms([(); 6].map(|_| {
                ecs.spawn((
                    circle.next().unwrap() * r * 0.3 + pen.pos,
                    Art::TripletuftedTerrorworm,
                    Wander::around(pen.pos, r),
                    Velocity::default(),
                    Innocence::Unwavering,
                    Phys::new().pushfoot_bighit(),
                    Health::full(1),
                ))
            }))
        }

        fn living<'a>(&'a self, g: &'a Game) -> impl Iterator<Item = hecs::Entity> + 'a {
            self.0.iter().copied().filter(move |&e| !g.dead(e))
        }

        fn dead<'a>(&'a self, g: &'a Game) -> impl Iterator<Item = hecs::Entity> + 'a {
            self.0.iter().copied().filter(move |&e| g.dead(e))
        }

        fn guides<'a>(&'a self, g: &'a Game) -> impl Iterator<Item = (Art, Vec2)> + 'a {
            self.living(g).map(move |e| (Art::Sword, g.pos(e)))
        }

        /// Note: now all of the entities are invalid, only do this
        /// when all the existing ones are dead, you want 6 more, and
        /// you won't be fiddling with them after this.
        fn refill(&self, g: &mut Game, pen: Pen) {
            Self::new(&mut g.ecs, pen);
        }
    }
    let terrorworms = Terrorworms::new(&mut ecs, pen);

    tasks.append(&mut vec![
        Task {
            label: "Talk to NPC 2",
            req: Box::new(move |g| g.hero_dist_ent(npc2) < 1.3),
            guide: Box::new(npc2_guide),
            on_finish: Box::new(move |g| {
                if let Ok(mut npc) = g.ecs.get_mut::<Npc>(npc2) {
                    npc.ui(|ui, _| {
                        paragraph(
                            ui,
                            concat!(
                                "Oh, adventurer, \n",
                                "I'm so glad you've come along! \n",
                                "In fact, it's almost like I've \n",
                                "just been standing here since \n",
                                "the dawn of time itself, waiting \n",
                                "for someone to come help me. In fact, \n",
                                "it's almost as if I was created simply \n",
                                "to give you something to do. \n",
                                "\n",
                                "In this pen are the last \n",
                                "Tripletufted Terrorworms known to exist, \n",
                                "and I need you to forgo your conscience \n",
                                "and simply slaughter them: no remorse."
                            ),
                        )
                    });
                }
                pen.gate_open(&mut g.ecs)
            }),
            ..Default::default()
        },
        Task {
            label: "Enter the Pen",
            req: Box::new(move |g| g.hero_dist(pen_pos) < pen_radius * 0.8),
            guide: Box::new(move |_, gi| gi.push((Art::Arrow, pen_pos))),
            on_finish: Box::new(move |g| {
                for &tw in &terrorworms.0 {
                    drop(g.innocent_for(tw, 1));
                }
                pen.gate_close(&mut g.ecs)
            }),
            ..Default::default()
        },
        Task {
            label: "Slaughter the first one",
            req: Box::new(move |g| terrorworms.dead(g).count() == 1),
            guide: Box::new(move |g, gi| gi.extend(terrorworms.guides(g))),
            on_finish: Box::new(move |g| {
                for &tw in &terrorworms.0 {
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
            label: "Shit boutta get real",
            req: Box::new(move |g| terrorworms.dead(g).count() == 2),
            guide: Box::new(move |g, gi| gi.extend(terrorworms.guides(g))),
            on_finish: Box::new(move |g| {
                for &tw in &terrorworms.0 {
                    drop(g.innocent_for(tw, 10));
                    drop(g.ecs.remove_one::<Wander>(tw));
                    drop(g.ecs.insert(
                        tw,
                        (
                            Fighter {
                                chase_speed: 0.0035,
                                charge_speed: 0.0055,
                                attack_interval: 140,
                                aggroed: true,
                                ..Default::default()
                            },
                            Bag::holding(Item::sword(tw)),
                            Health::full(2),
                            HealthBar,
                        ),
                    ));
                }
            }),
            ..Default::default()
        },
        Task {
            label: "Kill them all!",
            req: Box::new(move |g| terrorworms.living(g).count() == 0),
            on_finish: Box::new(move |g| pen.gate_open(&mut g.ecs)),
            ..Default::default()
        },
        Task {
            label: "Exit the Pen",
            req: Box::new(move |g| g.hero_dist(pen_pos) > pen_radius * 1.2),
            guide: Box::new(npc2_guide),
            on_finish: Box::new({
                use Trinket::*;
                #[rustfmt::skip]
                let mut reward = RewardChoice::new(hash!(), &[
                    (Item::Trinket(VolcanicShard), 1, "Volcanic Shard", None),
                    (Item::Trinket(LoftyInsoles), 1, "Lofty Insoles", None),
                ]);

                move |g| {
                    terrorworms.refill(g, pen);
                    pen.gate_close(&mut g.ecs);
                    if let Ok(mut npc) = g.ecs.get_mut::<Npc>(npc2) {
                        npc.ui(move |ui, g| {
                            paragraph(
                                ui,
                                concat!(
                                    "Your admirable conduct has confirmed our \n",
                                    "prophecies! Please, take a trinket!\n",
                                ),
                            );

                            for item in reward.ui(ui).into_iter() {
                                or_err!(g.give_item(hero, item));
                            }

                            paragraph(
                                ui,
                                concat!(
                                    "The slaughtered Tripletufted Terrorworms \n",
                                    "have simply returned. They will continue \n",
                                    "to grow in number until this pen, \n",
                                    "and eventually our simple way of life, \n",
                                    "is consumed by them. It is fortold that the \n",
                                    "only way we can stop them is by forging the \n",
                                    "Honeycoated Heptahorn.\n",
                                ),
                            );
                        });
                    }
                }
            }),
            ..Default::default()
        },
    ]);

    quests.add(Quest {
        title: "RPG Tropes I",
        completion_quip: "Try not to poke your eye out, like the last adventurer ...",
        completion: QuestCompletion::Accepted { on_tick: 0 },
        tasks,
        ..Default::default()
    });

    const STEP_EVERY: f64 = 1.0 / 60.0;
    let mut time = STEP_EVERY;
    let mut step = get_time();
    let mut game = Game::new(ecs);
    loop {
        time += get_time() - step;
        step = get_time();
        while time >= STEP_EVERY {
            time -= STEP_EVERY;
            game.tick = game.tick.wrapping_add(1);
            game.mouse_pos = drawer.cam.screen_to_world(mouse_position().into());
            if !is_mouse_button_down(MouseButton::Left) {
                game.mouse_down_tick = None;
            }
            if is_mouse_button_down(MouseButton::Left) && game.mouse_down_tick.is_none() {
                game.mouse_down_tick = Some(game.tick);
            }

            physics.tick(&mut game.ecs);
            quests.update(&mut game);
            quests.update_guides(&mut game, &drawer);
            bag_ui.update_icon(&mut game, &drawer);
            waffle.update(&mut game);
            keeper_of_bags.keep(&mut game.ecs);
            fireballs.update(&mut game, &mut quests);
            wander(&mut game.ecs);

            update_hero(hero, &mut game, &mut quests, &mut bag_ui);

            drawer.update(&mut game);
        }

        drawer.draw(&game);
        if let Ok(&hp) = game.ecs.get::<Health>(hero).as_deref() {
            let screen = vec2(screen_width(), screen_height());
            let size = vec2(screen.x() * (1.0 / 6.0), 30.0);
            health_bar(size, hp.ratio(), vec2(size.x() / 2.0 + 30.0, 70.0));
        }

        quests.ui(&game);
        npc_ui.ui(&mut game);
        if let Ok(bag) = game.ecs.get_mut(hero).as_deref_mut() {
            bag_ui.ui(bag);
        }
        megaui_macroquad::draw_megaui();

        fps.update();
        fps.draw();

        next_frame().await
    }
}

struct Game {
    ecs: hecs::World,
    hero_pos: Vec2,
    /// Any stage of swinging
    hero_swinging: bool,
    /// In game coordinates
    mouse_pos: Vec2,
    mouse_down_tick: Option<u32>,
    tick: u32,
}
impl Game {
    fn new(ecs: hecs::World) -> Self {
        Self {
            ecs,
            hero_pos: Vec2::zero(),
            hero_swinging: false,
            tick: 0,
            mouse_pos: Vec2::zero(),
            mouse_down_tick: None,
        }
    }

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

    fn give_item(&mut self, e: hecs::Entity, item: Item) -> Result<(), hecs::ComponentError> {
        self.ecs.get_mut::<Bag>(e)?.take(item);
        Ok(())
    }

    fn pos(&self, e: hecs::Entity) -> Vec2 {
        self.ecs.get::<Vec2>(e).as_deref().copied().unwrap_or_default()
    }

    /// The tuple's first field is true if they died,
    /// the second field represents knockback.
    fn hit(
        &mut self,
        HitInput { hitter_pos, hit: WeaponHit { hit }, mut damage, knockback, crit_chance }: HitInput,
    ) -> (bool, Vec2) {
        let tick = self.tick;

        if rand::gen_range(0.0, 1.0) < crit_chance {
            damage *= 3;
        }

        if let Ok((Health(hp, _), vel, &pos, ino, fighter)) = self.ecs.query_one_mut::<(
            &mut _,
            Option<&mut Velocity>,
            &Vec2,
            Option<&Innocence>,
            Option<&mut Fighter>,
        )>(hit)
        {
            if let Some(true) = ino.map(|i| i.active(tick)) {
                return (false, Vec2::zero());
            }

            if let Some(fighter) = fighter {
                fighter.aggroed = true;
            }

            if let Some(v) = vel {
                v.knockback(pos - hitter_pos, knockback);
            }

            *hp = hp.saturating_sub(damage);
            let dead = *hp == 0;
            if dead {
                or_err!(self.ecs.despawn(hit));
            }
            self.ecs.spawn((DamageLabel { tick, hp: -(damage as i32), pos },));

            (dead, (hitter_pos - pos).normalize())
        } else {
            (false, Vec2::zero())
        }
    }
}
struct HitInput {
    hitter_pos: Vec2,
    hit: WeaponHit,
    crit_chance: f32,
    damage: u32,
    knockback: f32,
}

struct Task {
    label: &'static str,
    req: Box<dyn FnMut(&Game) -> bool>,
    guide: Box<dyn FnMut(&Game, &mut Vec<(Art, Vec2)>)>,
    on_finish: Box<dyn FnMut(&mut Game)>,
    finished: bool,
}
impl Default for Task {
    fn default() -> Self {
        Self {
            label: "",
            req: Box::new(|_| true),
            on_finish: Box::new(|_| {}),
            guide: Box::new(|_, _| {}),
            finished: false,
        }
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
    completion_quip: &'static str,
    tasks: Vec<Task>,
    unlocks: Vec<usize>,
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

    #[allow(dead_code)]
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
    temp: Vec<usize>,
    tab_titles: [String; 3],
    new_tabs: [bool; 3],
    jump_to_tab: Option<usize>,
    window_open: bool,
    guides: Vec<(Art, Vec2)>,
    guide_ents: Vec<hecs::Entity>,
    icon_ent: hecs::Entity,
}
impl Quests {
    fn new(ecs: &mut hecs::World) -> Self {
        Self {
            quests: QuestVec(Vec::with_capacity(100)),
            temp: Vec::with_capacity(100),
            tab_titles: [(); 3].map(|_| String::with_capacity(25)),
            new_tabs: [false; 3],
            jump_to_tab: None,
            window_open: false,
            guide_ents: Vec::with_capacity(100),
            guides: Vec::with_capacity(100),
            icon_ent: ecs.spawn((vec2(0.0, 0.0), Art::Compass, ZOffset(-999.0))),
        }
    }

    fn add(&mut self, q: Quest) -> usize {
        let i = self.quests.0.len();
        self.quests.0.push(q);
        i
    }

    fn update(&mut self, g: &mut Game) {
        let Self { quests, guides, jump_to_tab, temp, new_tabs, .. } = self;

        guides.drain(..);
        for (_, _, quest) in quests.accepted_mut() {
            if let Some(task) =
                quest.tasks.iter_mut().find_map(|t| if !t.done(g) { Some(t) } else { None })
            {
                (task.guide)(g, guides);
            } else {
                quest.completion.finish(g.tick);
                *jump_to_tab = Some(2);
                if !self.window_open {
                    new_tabs[2] = true;
                }
                temp.extend(quest.unlocks.iter().copied());
            }
        }

        for unlock in temp.drain(..) {
            new_tabs[0] = true;
            quests.0[unlock].completion.unlock();
        }
    }

    fn update_guides(&mut self, g: &mut Game, drawer: &Drawer) {
        let Self { guides, guide_ents, .. } = self;
        for (_, &e) in guide_ents.iter().enumerate().skip_while(|&(i, _)| i < guides.len()) {
            drop(g.ecs.remove::<(Art, Vec2)>(e));
        }

        let screen = drawer.screen;
        let top_left = drawer.cam.target - screen;
        let flip_y = vec2(1.0, -1.0);
        for (i, (art, goal)) in guides.drain(..).enumerate() {
            let ent = guide_ents.get(i).copied().unwrap_or_else(|| {
                let e = g.ecs.reserve_entity();
                guide_ents.push(e);
                e
            });
            let from = (goal - top_left) * flip_y;
            let screen_size = screen * flip_y * 2.0;
            let (mut pos, rot, scale) =
                if from.cmplt(screen_size).all() && from.cmpgt(Vec2::zero()).all() {
                    (goal + vec2(0.0, 1.0), Rot(PI), 1.0)
                } else {
                    (
                        top_left + from.max(Vec2::zero()).min(screen_size) * flip_y,
                        Rot(Rot::from_vec2(drawer.cam.target - goal).0 + FRAC_PI_2),
                        1.35,
                    )
                };

            let bob = math::smoothstep((g.tick as f32 / 10.0).sin()) * 0.2;
            let v = Rot(rot.0 - FRAC_PI_2).vec2();
            pos += (v * 0.67 * scale) + (v * bob);
            or_err!(g
                .ecs
                .insert(ent, (art, pos, rot, ZOffset(-999.0), Scale(vec2(1.0, 0.4) * scale))));
        }

        if let Ok(mut pos) = g.ecs.get_mut::<Vec2>(self.icon_ent) {
            let bob = if self.new_tabs.iter().any(|&b| b) {
                (g.tick as f32 / 8.0).sin().abs() * 0.25
            } else {
                0.0
            };
            *pos = drawer.cam.target - screen * flip_y + vec2(0.5, 0.1 + bob);

            if is_key_pressed(KeyCode::Q)
                || g.mouse_down_tick == Some(g.tick)
                    && (*pos + vec2(0.0, 0.5) - g.mouse_pos).length() < 0.5
            {
                self.window_open = !self.window_open;
            }
        }
    }

    fn unlocked_ui(&mut self, ui: &mut megaui::Ui, tick: u32) {
        ui.separator();
        for (_, quest) in self.quests.unlocked_mut() {
            ui.label(None, quest.title);
            if ui.button(None, "Accept") {
                quest.completion.accept(tick);
            }
            ui.separator();
        }
    }

    fn accepted_ui(&mut self, ui: &mut megaui::Ui) {
        ui.separator();
        for (_, i, quest) in self.quests.accepted() {
            open_tree(ui, hash!(i), quest.title, |ui| {
                for task in &quest.tasks {
                    ui.label(None, &task.label);
                    if !task.finished {
                        return;
                    }
                }
            });
            ui.separator();
        }
    }

    fn finished_ui(&mut self, ui: &mut megaui::Ui) {
        ui.separator();
        for (i, _, quest) in self.quests.finished() {
            open_tree(ui, hash!(i), quest.title, |ui| {
                ui.label(None, quest.completion_quip);
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
                widgets::{Group, Tabbar},
                Vector2,
            },
            WindowParams,
        };

        if !self.window_open {
            return;
        }

        let size = Self::size();
        self.window_open = draw_window(
            hash!(),
            vec2(screen_width(), screen_height()) / 2.0 - size * vec2(0.5, -0.1),
            size,
            WindowParams {
                label: "Quests - toggle with Q".to_string(),
                close_button: true,
                ..Default::default()
            },
            |ui| {
                let tab_height = 22.5;
                let tab = {
                    let jump = self.jump_to_tab.take();
                    let titles = self.tab_titles();
                    Tabbar::new(
                        hash!(),
                        Vector2::new(0.0, 0.0),
                        Vector2::new(size.x(), tab_height),
                        &titles,
                    )
                    .selected_tab(jump)
                    .ui(ui)
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
        draw_text(&self.average().to_string(), screen_width() - 80.0, 0.0, 30.0, BLACK);
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
            draw_text(
                text,
                x - 20.0,
                y - 120.0 - (game_tick - tick) as f32,
                42.0 * (1.0 + ((hp as f32).abs() - 1.0) / 3.0),
                MAROON,
            );
        }
    }
}

fn health_bar(size: Vec2, ratio: f32, pos: Vec2) {
    fn lerp_color(Color(a): Color, Color(b): Color, t: f32) -> Color {
        pub fn lerp(a: u8, b: u8, t: f32) -> u8 {
            a + (((b - a) as f32 / 255.0 * t) * 255.0) as u8
        }
        Color([lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)])
    }

    let (x, y) = (pos - size * vec2(0.5, 1.4)).into();
    let (w, h) = size.into();
    let mut color =
        if ratio < 0.5 { lerp_color(RED, YELLOW, ratio) } else { lerp_color(YELLOW, GREEN, ratio) };
    color.0[3] = 150;
    draw_rectangle_lines(x, y, w, h, size.y() * 0.3, color);
    color.0[3] = 100;
    draw_rectangle(x, y, w * ratio, h, color);
}

type SpriteData = (Vec2, Art, ZOffset, Option<Rot>, Option<Scale>);
struct Drawer {
    sprites: Vec<SpriteData>,
    damage_labels: DamageLabelBin,
    cam: Camera2D,
    screen: Vec2,
}
impl Drawer {
    fn new() -> Self {
        Self {
            sprites: Vec::with_capacity(1000),
            cam: Default::default(),
            screen: Vec2::zero(),
            damage_labels: DamageLabelBin::new(),
        }
    }

    fn update(&mut self, game: &mut Game) {
        let Self { damage_labels, cam, .. } = self;
        damage_labels.update(game);
        cam.target = cam.target.lerp(game.hero_pos + vec2(0.0, 0.5), 0.05);
    }

    fn draw(&mut self, game: &Game) {
        self.sprites(game);

        self.damage_labels.draw(game, &self.cam);
    }

    fn sprites(&mut self, Game { ecs, tick, .. }: &Game) {
        let tick = *tick;
        let Self { screen, sprites, cam, .. } = self;
        cam.zoom = vec2(1.0, screen_width() / screen_height()) / 7.8;
        set_camera(*cam);
        *screen = cam.screen_to_world(vec2(screen_width(), screen_height())) - cam.target;
        let top_left = cam.target - *screen;

        #[cfg(feature = "show-culling")]
        {
            cam.zoom = vec2(1.0, screen_width() / screen_height()) / 7.8 / 2.0;
            set_camera(*cam);
        }

        clear_background(Color([180, 227, 245, 255]));
        fn road_through(p: Vec2, slope: Vec2) {
            let ((sx, sy), (ex, ey)) = ((-slope + p).into(), (slope + p).into());
            draw_line(sx, sy, ex, ey, 3.2, Color([160, 160, 160, 255]));
        }
        draw_line(-20.0, -20.0, -40.0, 10.0, 2.2, Color([165, 165, 165, 255]));
        road_through(-vec2(20.0, 20.0), vec2(180.0, 420.0));

        #[cfg(feature = "show-culling")]
        {
            let (x, y) = (cam.target - screen).into();
            let (w, h) = (screen * 2.0).into();
            draw_rectangle_lines(x, y, w, h, 0.1, RED);
        }

        let gl = unsafe { get_internal_gl().quad_gl };

        let flip_y = vec2(1.0, -1.0);
        let screen_size = *screen * flip_y * 2.0;

        sprites.extend(
            ecs.query::<(&_, &_, Option<&ZOffset>, Option<&Rot>, Option<&Scale>)>()
                .iter()
                .map(|(_, (&p, &a, z, r, s))| {
                    (p, a, z.copied().unwrap_or_default(), r.copied(), s.copied())
                })
                .filter(|&(p, art, _, rot, _): &SpriteData| {
                    let (x, m) = rot.map(|_| (0.0, 2.0)).unwrap_or((1.0, 1.0));
                    let bound = Vec2::splat(art.bounding() * m);
                    let from = (p + vec2(0.0, bound.x() * x) - top_left) * flip_y;
                    from.cmplt(screen_size + bound).all() && from.cmpgt(-bound).all()
                }),
        );
        sprites.sort_by(|a, b| float_cmp(b, a, |(pos, art, z, ..)| pos.y() + art.z_offset() + z.0));
        for (pos, art, _, rot, scale) in sprites.drain(..) {
            let mut push_model_matrix = || {
                gl.push_model_matrix(
                    glam::Mat4::from_translation(glam::vec3(pos.x(), pos.y(), 0.0))
                        * rot.map(|Rot(r)| glam::Mat4::from_rotation_z(r)).unwrap_or_default()
                        * scale
                            .map(|Scale(v)| glam::Mat4::from_scale(glam::vec3(v.x(), v.y(), 1.0)))
                            .unwrap_or_default(),
                );
            };

            let (x, y) = pos.into();
            let rect = |color: Color, w: f32, h: f32| {
                draw_rectangle(x - w / 2.0, y, w, h, color);
            };

            match art {
                Art::Hero => rect(BLUE, 1.0, 1.0),
                Art::Scarecrow => rect(GOLD, 0.8, 0.8),
                Art::TripletuftedTerrorworm => rect(WHITE, 0.8, 0.8),
                Art::VioletVagabond => rect(VIOLET, 0.8, 0.8),
                Art::Npc => rect(GREEN, 1.0, GOLDEN_RATIO),
                Art::Chest => {
                    let (w, h) = (vec2(GOLDEN_RATIO, 1.0) * 0.8).into();
                    draw_rectangle(x - w / 2.0, y, w, h, BROWN);
                    draw_line(x - w / 2.0, y + 0.435, x + w / 2.0, y + 0.435, 0.05, DARKGRAY);
                    draw_rectangle_lines(x - w / 2.0, y, w, h, 0.1, DARKGRAY);
                    draw_circle(x, y + h / 2.0, 0.10, GOLD);
                    draw_circle(x, y + h / 2.0, 0.06, DARKGRAY);
                    draw_rectangle(x - 0.1, y + h / 2.0 - 0.135, 0.2, 0.135, GOLD);
                }
                Art::Compass => {
                    draw_circle(x, y + 0.40, 0.4000, DARKGRAY);
                    draw_circle(x, y + 0.40, 0.3055, GRAY);
                    draw_triangle(
                        vec2(x - 0.085, y + 0.275),
                        vec2(x + 0.085, y + 0.275),
                        vec2(x + 0.025, y + 0.65),
                        LIGHTGRAY,
                    );
                    draw_triangle(
                        vec2(x - 0.085, y + 0.275),
                        vec2(x + 0.085, y + 0.275),
                        vec2(x, y + 0.185),
                        LIGHTGRAY,
                    );
                }
                Art::Lockbox => {
                    let (w, h) = (0.8, 0.435);
                    draw_circle(x, y + h, 0.4, DARKGRAY);
                    draw_circle(x, y + h, 0.4 - 0.0945, GRAY);
                    draw_rectangle(x - w / 2.0, y, w, h, GRAY);
                    draw_rectangle_lines(x - w / 2.0, y, w, h, 0.0945 * 2.0, DARKGRAY);
                    let lock = 1.35;
                    draw_circle(x, y + h - 0.08, 0.10 * lock, LIGHTGRAY);
                    draw_circle(x, y + h - 0.08, 0.06 * lock, DARKGRAY);
                    draw_rectangle(
                        x - 0.1 * lock,
                        y + h - 0.201 * lock,
                        0.2 * lock,
                        0.135 * lock,
                        LIGHTGRAY,
                    );
                }
                Art::Tree => {
                    let (w, h, r) = (0.8, GOLDEN_RATIO * 0.8, 0.4);
                    draw_circle(x, y + r, r, BROWN);
                    draw_rectangle(x - w / 2.0, y + r, w, h, BROWN);
                    draw_circle(x + 0.80, y + 2.2, 0.8, DARKGREEN);
                    draw_circle(x + 0.16, y + 3.0, 1.0, DARKGREEN);
                    draw_circle(x - 0.80, y + 2.5, 0.9, DARKGREEN);
                    draw_circle(x - 0.16, y + 2.0, 0.8, DARKGREEN);
                }
                Art::Post => {
                    let (w, h, r) = (0.8, GOLDEN_RATIO * 0.8, 0.4);
                    draw_circle(x, y + r, r, BROWN);
                    draw_rectangle(x - w / 2.0, y + r, w, h, BROWN);
                    draw_circle(x, y + h + r, 0.4, BEIGE);
                }
                Art::Fireball => {
                    push_model_matrix();
                    let mut color = ORANGE;
                    for q in 0..10 {
                        let i = q % 5;
                        let w = (5 - i) as f32 / 5.0 * 0.4 + 0.1;
                        color.0[3] = 60 + 30 * ((5 - i) as u8 / 5);
                        draw_rectangle(
                            w / -2.0 + i as f32 / 6.0 + q as f32 / 85.0,
                            ((tick + q * 3) as f32 / 5.0).sin() * (i as f32 / 4.5).max(0.05) * 0.65
                                - w / 2.0,
                            w,
                            w,
                            color,
                        );
                    }
                    color = RED;
                    color.0[3] = 100;
                    draw_rectangle(
                        0.3 / -2.0,
                        0.3 / -2.0 + (tick as f32 / 5.0).sin() * 0.02,
                        0.3,
                        0.3,
                        color,
                    );
                    gl.pop_model_matrix();
                }
                Art::Arrow => {
                    push_model_matrix();
                    draw_triangle(vec2(0.11, 0.0), vec2(-0.11, 0.0), vec2(0.00, GOLDEN_RATIO), RED);
                    draw_triangle(
                        vec2(-0.225, 0.85),
                        vec2(0.225, 0.85),
                        vec2(0.00, GOLDEN_RATIO),
                        RED,
                    );
                    gl.pop_model_matrix();
                }
                Art::Sword => {
                    push_model_matrix();
                    draw_triangle(
                        vec2(0.075, 0.0),
                        vec2(-0.075, 0.0),
                        vec2(0.00, GOLDEN_RATIO),
                        DARKBROWN,
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
                    gl.pop_model_matrix()
                }
                Art::Consumable(_) | Art::Trinket(_) => {}
            };
        }

        system!(ecs, _,
            &hp        = &Health
            &HealthBar = &_
            &pos       = &Vec2
        {
            health_bar(vec2(1.0, 0.2), hp.ratio(), pos);
        });

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

        #[cfg(feature = "show-culling")]
        system!(ecs, _,
            &art = &Art
            &pos = &Vec2
            rot  = Option<&Rot>
        {
            let mut color = BEIGE;
            color.0[3] = 100;
            let l = art.bounding() * rot.map(|_| 4.0).unwrap_or(2.0);
            let (x, y) = pos.into();
            draw_rectangle(x - l / 2.0, rot.map(|_| y - l / 2.0).unwrap_or(y), l, l, color);
        });

        set_default_camera();
    }
}
