use std::mem::swap;

use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};

const WORLD_SIZE: u32 = 64;
const DISPLAY_SIZE: u32 = 2048;
const SCALE: u32 = DISPLAY_SIZE / WORLD_SIZE;

// https://github.com/markjarzynski/PCG3D/blob/master/pcg3d.hlsl
#[tracked]
pub fn pcg3d(v: Expr<Vec3<u32>>) -> Expr<Vec3<u32>> {
    let v = v.var();
    *v = v * 1664525u32 + 1013904223u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    *v ^= v >> 16u32;

    *v.x += v.y * v.z;
    *v.y += v.z * v.x;
    *v.z += v.x * v.y;

    **v
}

#[tracked]
pub fn pcg3df(v: Expr<Vec3<u32>>) -> Expr<Vec3<f32>> {
    pcg3d(v).cast_f32() / u32::MAX as f32
}

fn opposite(i: u32) -> u32 {
    (i + 2) % 4
}
fn negative(i: u32) -> bool {
    i < 2
}
fn axis(i: u32) -> u32 {
    i % 2
}

fn in_dir(v: Expr<Vec4<f32>>, i: Expr<u32>) -> Expr<f32> {
    let v = Expr::<[f32; 4]>::from(v);
    v.read(i)
}
fn in_dir_var(v: Var<Vec4<f32>>, i: u32) -> Var<f32> {
    match i {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        3 => v.w,
        _ => unreachable!(),
    }
}

fn directions() -> [Vec2<i32>; 4] {
    [
        Vec2::new(-1, 0),
        Vec2::new(0, -1),
        Vec2::new(1, 0),
        Vec2::new(0, 1),
    ]
}
fn offset() -> [Vec2<u32>; 4] {
    [
        Vec2::new(0, 0),
        Vec2::new(0, 0),
        Vec2::new(SCALE - 1, 0),
        Vec2::new(0, SCALE - 1),
    ]
}
fn perp_axis() -> [Vec2<u32>; 4] {
    [Vec2::y(), Vec2::x(), Vec2::y(), Vec2::x()]
}

struct VectorField {
    // All values are pointing outwards.
    values: Tex2d<Vec4<f32>>,
}
impl VectorField {
    fn new() -> Self {
        Self {
            values: DEVICE.create_tex2d(PixelStorage::Float4, WORLD_SIZE, WORLD_SIZE, 1),
        }
    }
    #[tracked]
    fn at(values: Tex2dVar<Vec4<f32>>, pos: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        let cell = pos.cast_u32();
        let frac = pos - cell.cast_f32();
        let cell = cell.clamp(0, WORLD_SIZE - 1);
        let value = values.read(cell);
        Vec2::expr(
            value.x * (frac.x - 1.0) + value.z * frac.x,
            value.y * (frac.y - 1.0) + value.w * frac.y,
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
struct PortalEndpoint {
    velocity: f32,
    cell: Vec2<u32>,
    dir: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
struct Portal {
    endpoints: [PortalEndpoint; 2],
}

struct World {
    portals: Buffer<Portal>,
    // First bit is which end point.
    world_portals: Tex2d<Vec4<u32>>,
}

fn main() {
    let app = App::new("Mobius", [WORLD_SIZE; 2]).scale(SCALE).init();

    let mut portals = vec![];
    for x in WORLD_SIZE / 4..3 * WORLD_SIZE / 4 {
        portals.push(Portal {
            endpoints: [
                PortalEndpoint {
                    velocity: 0.0,
                    cell: Vec2::new(x, WORLD_SIZE / 4),
                    dir: 1,
                },
                PortalEndpoint {
                    velocity: 0.0,
                    cell: Vec2::new(x, 3 * WORLD_SIZE / 4),
                    dir: 3,
                },
            ],
        });
        portals.push(Portal {
            endpoints: [
                PortalEndpoint {
                    velocity: 0.0,
                    cell: Vec2::new(x, WORLD_SIZE / 4 - 1),
                    dir: 3,
                },
                PortalEndpoint {
                    velocity: 0.0,
                    cell: Vec2::new(x, 3 * WORLD_SIZE / 4 + 1),
                    dir: 1,
                },
            ],
        });
    }

    let world = World {
        portals: DEVICE.create_buffer_from_slice(&portals),
        world_portals: DEVICE.create_tex2d(PixelStorage::Int4, WORLD_SIZE, WORLD_SIZE, 1),
    };

    let reset_world = DEVICE.create_kernel::<fn()>(&track!(|| {
        let cell = dispatch_id().xy();
        world.world_portals.write(cell, Vec4::splat(u32::MAX));
    }));
    let update_world = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let portal = world.portals.read(index);
        for i in (0..2_u32) {
            let endpoint = portal.endpoints.read(i);
            let cell = endpoint.cell;
            let world_cell = Expr::<[_; 4]>::from(world.world_portals.read(cell)).var();
            world_cell.write(endpoint.dir, index * 2 + (1 - i));
            world
                .world_portals
                .write(cell, Expr::<Vec4<_>>::from(**world_cell));
        }
    }));
    reset_world.dispatch([WORLD_SIZE, WORLD_SIZE, 1]);
    update_world.dispatch([world.portals.len() as u32, 1, 1]);

    let mut gravity = VectorField::new();
    let mut staging = DEVICE.create_tex2d(PixelStorage::Float4, WORLD_SIZE, WORLD_SIZE, 1);

    let divergence_solve = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>)>(&track!(
        |field, next_field| {
            let cell = dispatch_id().xy();
            let divergence = field.read(cell).var();
            *divergence -= divergence.reduce_sum() / 4.0 * 1.9;
            next_field.write(cell, divergence);
        }
    ));
    let realign = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>, Vec2<f32>)>(
        &track!(|field, next_field, boundary| {
            let cell = dispatch_id().xy();
            let portals = world.world_portals.read(cell);
            let divergence = field.read(cell).var();
            for i in (0..4_u32) {
                let dir = directions()[i as usize];
                let v = in_dir_var(divergence, i);
                let neighbor = (cell + dir.expr().cast_u32()).var();
                let neighbor_dir = opposite(i).var();
                let portal = Expr::<[u32; 4]>::from(portals).read(i);
                if portal != u32::MAX {
                    let other_endpoint = world.portals.read(portal / 2).endpoints.read(portal % 2);
                    *neighbor = other_endpoint.cell;
                    *neighbor_dir = other_endpoint.dir;
                }
                if (neighbor >= WORLD_SIZE).any() {
                    *v = if negative(i) { -1.0 } else { 1.0 }
                        * if axis(i) == 0 { boundary.x } else { boundary.y };
                } else {
                    if portal != u32::MAX
                        || Expr::<[u32; 4]>::from(world.world_portals.read(**neighbor))
                            .read(neighbor_dir)
                            == u32::MAX
                    {
                        let neighbor = field.read(neighbor);
                        let n = -in_dir(neighbor, **neighbor_dir);
                        *v = (v + n) * 0.5;
                    }
                }
            }
            next_field.write(cell, divergence);
        }),
    );

    let tracers = DEVICE.create_buffer_from_fn::<Vec2<f32>>(1000, |_| Vec2::splat(-1.0));
    let tracer_display = DEVICE.create_tex2d(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

    let draw = DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>)>(&track!(|field| {
        let cell = dispatch_id().xy();
        let pos = cell.cast_f32() + 0.5;
        let v = VectorField::at(field, pos);
        let colors = [
            Vec3::new(0.64178, 0.22938, 0.33132), // 0
            Vec3::new(0.06965, 0.44936, 0.3549),  // 180
            Vec3::new(0.47086, 0.33081, 0.08135), // 90
            Vec3::new(0.23872, 0.32924, 0.72414), // 270
        ];
        app.set_pixel(
            cell.cast_i32(),
            keter::max(v.x, 0.0) * colors[0]
                + keter::max(-v.x, 0.0) * colors[1]
                + keter::max(v.y, 0.0) * colors[2]
                + keter::max(-v.y, 0.0) * colors[3],
        );
        let portals = world.world_portals.read(cell);
        for i in (0..4_u32) {
            if Expr::<[u32; 4]>::from(portals).read(i) != u32::MAX {
                let overlay_pos = cell * SCALE + offset()[i as usize];
                for j in 0..SCALE {
                    let overlay_pos = overlay_pos + perp_axis()[i as usize] * j;
                    app.overlay().write(overlay_pos, Vec4::splat(1.0));
                }
            }
        }
    }));

    let update_tracers =
        DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, u32)>(&track!(|gravity, t| {
            let tracer = tracers.read(dispatch_id().x).var();
            if (tracer <= 0.01).any() || (tracer >= WORLD_SIZE as f32 - 0.01).any() {
                *tracer = pcg3df(Vec3::expr(dispatch_id().x, 153, t)).xy() * WORLD_SIZE as f32;
            }
            *tracer += VectorField::at(gravity, **tracer) * 0.1;
            tracers.write(dispatch_id().x, tracer);
            tracer_display.write(
                (tracer * (SCALE) as f32).cast_u32(),
                Vec4::splat(1.0).expr(),
            );
        }));
    let draw_tracers = DEVICE.create_kernel::<fn()>(&track!(|| {
        let pixel = dispatch_id().xy();
        let value = tracer_display.read(pixel).var();
        app.overlay().write(pixel, value);
        *value.w *= 0.995;
        tracer_display.write(pixel, value);
    }));

    app.run(|rt| {
        if rt.key_pressed(KeyCode::KeyR) {
            rt.begin_recording(None, true);
        } else if rt.key_pressed(KeyCode::KeyX) {
            rt.finish_recording();
        }
        divergence_solve.dispatch([WORLD_SIZE, WORLD_SIZE, 1], &gravity.values, &staging);
        swap(&mut gravity.values, &mut staging);
        realign.dispatch(
            [WORLD_SIZE, WORLD_SIZE, 1],
            &gravity.values,
            &staging,
            &Vec2::new(0.0, 1.0),
        );
        swap(&mut gravity.values, &mut staging);
        update_tracers.dispatch([tracers.len() as u32, 1, 1], &gravity.values, &rt.tick);
        draw_tracers.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
        draw.dispatch([WORLD_SIZE, WORLD_SIZE, 1], &gravity.values);
    });
}
