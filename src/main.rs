use std::mem::swap;

use keter::lang::types::vector::{Vec2, Vec3, Vec4};
use keter::prelude::*;
use keter_testbed::{App, KeyCode};

const WORLD_SIZE: u32 = 128;
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

#[tracked]
fn rotate(v: Expr<Vec2<f32>>, rotation: Expr<u32>) -> Expr<Vec2<f32>> {
    if rotation == 0 {
        v
    } else if rotation == 1 {
        Vec2::expr(-v.y, v.x)
    } else if rotation == 2 {
        Vec2::expr(-v.x, -v.y)
    } else {
        Vec2::expr(v.y, -v.x)
    }
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
    world_portals: Buffer<u32>,
}
impl World {
    #[tracked]
    fn read(&self, cell: Expr<Vec2<u32>>, dir: Expr<u32>) -> Expr<u32> {
        let index = 4 * (cell.x * WORLD_SIZE + cell.y) + dir;
        self.world_portals.read(index)
    }
    #[tracked]
    fn write(&self, cell: Expr<Vec2<u32>>, dir: Expr<u32>, value: Expr<u32>) {
        let index = 4 * (cell.x * WORLD_SIZE + cell.y) + dir;
        self.world_portals.write(index, value);
    }
    #[tracked]
    fn trace_line(
        &self,
        ray_start: Expr<Vec2<f32>>,
        ray_dir: Expr<Vec2<f32>>,
    ) -> (Expr<Vec2<f32>>, Expr<u32>) {
        let length = ray_dir.norm();
        let ray_dir = ray_dir / length;
        let inv_dir = (ray_dir + f32::EPSILON).recip();
        let pos = ray_start.floor().cast_i32().var();
        let delta_dist = inv_dir.abs();

        let ray_step = ray_dir.signum().cast_i32();
        let side_dist =
            (ray_dir.signum() * (pos.cast_f32() - ray_start) + ray_dir.signum() * 0.5 + 0.5)
                * delta_dist;
        let side_dist = side_dist.var();

        let rotation = 0_u32.var();

        for i in 0..10 {
            let next_t = side_dist.reduce_min();

            if next_t >= length {
                break;
            }
            let mask = side_dist <= side_dist.yx();

            *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));

            let step = mask.select(ray_step, Vec2::splat_expr(0));
            let step_dir = if step.x == -1 {
                0_u32.expr()
            } else if step.x == 1 {
                2.expr()
            } else if step.y == -1 {
                1.expr()
            } else {
                3.expr()
            };
            let step_dir = (step_dir + rotation) % 4;

            let portal = if (pos.cast_u32() < WORLD_SIZE).all() {
                self.read(pos.cast_u32(), step_dir)
            } else {
                u32::MAX.expr()
            };
            if portal != u32::MAX {
                let portal = self.portals.read(portal / 2).endpoints.read(portal % 2);
                *rotation = (rotation + portal.dir + (4 - step_dir) + 2) % 4;
                *pos = portal.cell.cast_i32();
            } else {
                *pos += step;
            }
        }
        let fract = (ray_dir * length + ray_start).fract();

        let fract = if rotation == 0 {
            fract
        } else if rotation == 1 {
            Vec2::expr(1.0 - fract.y, fract.x)
        } else if rotation == 2 {
            Vec2::expr(1.0 - fract.x, 1.0 - fract.y)
        } else {
            Vec2::expr(fract.y, 1.0 - fract.x)
        };

        (pos.cast_f32() + fract, **rotation)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Value)]
struct Object {
    pos: Vec2<f32>,
    vel: Vec2<f32>,
}

fn gen_portals(t: f32) -> Vec<Portal> {
    let mut portals = vec![];
    let p = ((WORLD_SIZE / 4) as f32 + 10.0 * (t * 0.016).sin()).round() as u32;
    let v = 10.0 * (t * 0.016).cos() * 0.016;
    for x in WORLD_SIZE / 4..3 * WORLD_SIZE / 4 {
        portals.push(Portal {
            endpoints: [
                PortalEndpoint {
                    velocity: v,
                    cell: Vec2::new(x, p),
                    dir: 1,
                },
                PortalEndpoint {
                    velocity: v,
                    cell: Vec2::new(x, 3 * WORLD_SIZE / 4 - 1),
                    dir: 3,
                },
            ],
        });
        portals.push(Portal {
            endpoints: [
                PortalEndpoint {
                    velocity: -v,
                    cell: Vec2::new(x, p - 1),
                    dir: 3,
                },
                PortalEndpoint {
                    velocity: -v,
                    cell: Vec2::new(x, 3 * WORLD_SIZE / 4),
                    dir: 1,
                },
            ],
        });
    }
    portals
}

fn main() {
    let app = App::new("Mobius", [WORLD_SIZE; 2]).scale(SCALE).init();

    let mut gravity = VectorField::new();
    let mut staging = DEVICE.create_tex2d(PixelStorage::Float4, WORLD_SIZE, WORLD_SIZE, 1);

    let world = World {
        portals: DEVICE.create_buffer_from_slice(&gen_portals(0.0)),
        world_portals: DEVICE
            .create_buffer_from_fn((WORLD_SIZE * WORLD_SIZE * 4) as usize, |_| u32::MAX),
    };

    let reset_world = DEVICE.create_kernel::<fn()>(&track!(|| {
        let cell = dispatch_id().xy();
        for i in 0..4_u32 {
            world.write(cell, i, u32::MAX.expr());
        }
    }));
    let update_world = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        let portal = world.portals.read(index);
        for i in (0..2_u32) {
            let endpoint = portal.endpoints.read(i);
            let cell = endpoint.cell;
            world.write(cell, endpoint.dir, index * 2 + (1 - i));
        }
    }));

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
            let divergence = field.read(cell).var();
            for i in (0..4_u32) {
                let dir = directions()[i as usize];
                let v = in_dir_var(divergence, i);
                let neighbor = (cell + dir.expr().cast_u32()).var();
                let neighbor_dir = opposite(i).var();
                let portal = world.read(cell, i.expr());
                let velocity = 0.0_f32.var();
                if portal != u32::MAX {
                    let other_endpoint = world.portals.read(portal / 2).endpoints.read(portal % 2);
                    *neighbor = other_endpoint.cell;
                    *neighbor_dir = other_endpoint.dir;
                    *velocity = other_endpoint.velocity;
                }
                if (neighbor >= WORLD_SIZE).any() {
                    *v = if negative(i) { -1.0 } else { 1.0 }
                        * if axis(i) == 0 { boundary.x } else { boundary.y };
                } else {
                    if portal != u32::MAX || world.read(**neighbor, **neighbor_dir) == u32::MAX {
                        let neighbor = field.read(neighbor);
                        let n = -(in_dir(neighbor, **neighbor_dir) + velocity);
                        *v = (v + n) * 0.5;
                    }
                }
            }
            next_field.write(cell, divergence);
        }),
    );

    let tracers = DEVICE.create_buffer_from_fn::<Object>(10, |_| Object {
        pos: Vec2::splat(-1.0),
        vel: Vec2::splat(0.0),
    });
    let tracer_display = DEVICE.create_tex2d(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

    let draw =
        DEVICE.create_kernel::<fn(Tex2d<Vec4<f32>>, bool)>(&track!(|field, show_divergence| {
            let cell = dispatch_id().xy();
            let pos = cell.cast_f32() + 0.5;
            app.set_pixel(
                cell.cast_i32(),
                if show_divergence {
                    let dv = field.read(cell).reduce_sum();
                    dv.abs() * 2.0 * Vec3::new(1.0, 0.0, 0.0)
                } else {
                    let v = VectorField::at(field.clone(), pos);
                    let colors = [
                        Vec3::new(0.64178, 0.22938, 0.33132), // 0
                        Vec3::new(0.06965, 0.44936, 0.3549),  // 180
                        Vec3::new(0.47086, 0.33081, 0.08135), // 90
                        Vec3::new(0.23872, 0.32924, 0.72414), // 270
                    ];
                    keter::max(v.x, 0.0) * colors[0]
                        + keter::max(-v.x, 0.0) * colors[1]
                        + keter::max(v.y, 0.0) * colors[2]
                        + keter::max(-v.y, 0.0) * colors[3]
                },
            );

            for i in (0..4_u32) {
                if world.read(cell, i.expr()) != u32::MAX {
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
            if (tracer.pos <= 0.01).any() || (tracer.pos >= WORLD_SIZE as f32 - 0.01).any() {
                *tracer.pos = pcg3df(Vec3::expr(dispatch_id().x, 153, t)).xy() * WORLD_SIZE as f32;
                *tracer.vel = (pcg3df(Vec3::expr(dispatch_id().x, 121, t)).xy() * 1.0 - 0.5) * 0.1;
            }
            // +=
            let (next_pos, rotation) = world.trace_line(
                **tracer.pos,
                (**tracer.vel + VectorField::at(gravity.clone(), **tracer.pos)), // * 0.016,
            );
            // TODO: This doesn't handle mirrored portals correctly.
            *tracer.vel = rotate(**tracer.vel, rotation);
            *tracer.pos = next_pos;
            tracers.write(dispatch_id().x, tracer);
            tracer_display.write(
                (tracer.pos * (SCALE) as f32).cast_u32(),
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

    let mut show_divergence = false;

    app.run(|rt| {
        if rt.key_pressed(KeyCode::KeyD) {
            show_divergence = !show_divergence;
        }
        rt.log_fps();

        // println!("Len: {:?}", gen_portals(rt.tick as f32 * 0.016).len());
        world.portals.copy_from(&gen_portals(rt.tick as f32));
        reset_world.dispatch([WORLD_SIZE, WORLD_SIZE, 1]);
        update_world.dispatch([world.portals.len() as u32, 1, 1]);

        for _ in 0..100 {
            divergence_solve.dispatch([WORLD_SIZE, WORLD_SIZE, 1], &gravity.values, &staging);
            swap(&mut gravity.values, &mut staging);
            realign.dispatch(
                [WORLD_SIZE, WORLD_SIZE, 1],
                &gravity.values,
                &staging,
                &Vec2::new(0.0, 0.0),
            );
            swap(&mut gravity.values, &mut staging);
        }
        update_tracers.dispatch([tracers.len() as u32, 1, 1], &gravity.values, &rt.tick);
        draw_tracers.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1]);
        draw.dispatch(
            [WORLD_SIZE, WORLD_SIZE, 1],
            &gravity.values,
            &show_divergence,
        );
    });
}
