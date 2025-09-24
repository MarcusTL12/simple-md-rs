use std::simd::{LaneCount, StdFloat, SupportedLaneCount, prelude::*};

use crate::f32::transpose::{transpose_nd, untranspose_nd};

#[inline(always)]
fn array_sum<const N: usize>(x: [f32; N], y: [f32; N]) -> [f32; N] {
    let mut ans = [0.0; N];

    for i in 0..N {
        ans[i] = x[i] + y[i];
    }

    ans
}

#[inline(always)]
fn array_diff<const N: usize>(x: [f32; N], y: [f32; N]) -> [f32; N] {
    let mut ans = [0.0; N];

    for i in 0..N {
        ans[i] = x[i] - y[i];
    }

    ans
}

// Returns sum(u_i * v_i)
#[inline(always)]
fn dot<const N: usize>(u: [f32; N], v: [f32; N]) -> f32 {
    let mut ans = 0.0;

    for (a, b) in u.into_iter().zip(v) {
        ans = a.mul_add(b, ans);
    }

    ans
}

// E = 4 * e * (k^2 - k) = 4 * e * k * (k - 1)
// k = (s/r)^6
// r = |r1 - r2|
#[inline(always)]
pub fn lj_energy<const N: usize>(
    r1: [f32; N],
    r2: [f32; N],
    sigma: f32,
    eps: f32,
) -> f32 {
    let r = array_diff(r1, r2);
    let rsq = dot(r, r);

    let sr2 = (sigma * sigma) / rsq;

    let k = sr2.powi(3);

    4.0 * eps * k * (k - 1.0)
}

// E = 4 * e * k * (k - 1)
// dE/dr2 = -dE/dr1
// dE/dr1 = dE/dk * dk/dr1
// dE/dk = 4 * e * (k + (k - 1))
//       = 4 * e * (2k - 1)
// k = u^3
// u = s^2 / r^2 = s^2 * r^-2 = s^2 * v^-1
// v = r^2
//
// dk/dr1 = dk/du * du/dv * dv/dr1
// dk/du = 3 * u^2
// du/dv = -s^2 * v^-2 = -s^2 * v^-1 * v^-1
//       = -u * v^-1
// dk/dr1 = -3 * k * v^-1 * dv/dr1
//
// v = r^2 = sum_i (r1_i - r2_i)^2
// dv/dr1_i = 2 (r1_i - r2_i) = r_i
// dk/dr1_i = -3 * k * v^-1 * r_i
//
// dE/dr1_i = -12 * e * (2k - 1) * k * v^-1 * r_i
//
// E = c * (k - 1)
// dE/dr1_i = -3 * c * (2k - 1) * v^-1 * r_i
// c = 4 * e * k
#[inline(always)]
fn lj_gradient<const N: usize>(
    r1: [f32; N],
    r2: [f32; N],
    sigma: f32,
    eps: f32,
) -> (f32, [f32; N]) {
    let r = array_diff(r1, r2);
    let v = dot(r, r);
    let vinv = 1.0 / v;

    let u = sigma * sigma * vinv;
    let k = u.powi(3);

    let c = 4.0 * eps * k;

    (
        c * (k - 1.0),
        r.map(|ri| -3.0 * c * (2.0 * k - 1.0) * vinv * ri),
    )
}

// Computes forces for all pairs of lennard jones particles.
// Returns total energy
pub fn compute_lj_forces<const N: usize>(
    coords: &[[f32; N]],
    forces: &mut [[f32; N]],
    sigma: f32,
    eps: f32,
) -> f32 {
    let n = coords.len();

    assert_eq!(forces.len(), n);

    for f in forces.iter_mut() {
        *f = [0.0; N];
    }

    let mut energy = 0.0;

    for i in 0..n {
        let r1 = coords[i];

        for j in 0..i {
            let r2 = coords[j];

            // g = dE/dr1
            // F1 = -dE/dr1
            // F2 = -dE/dr2 = dE/dr1
            let (e, grad) = lj_gradient(r1, r2, sigma, eps);

            energy += e;

            forces[i] = array_diff(forces[i], grad);
            forces[j] = array_sum(forces[j], grad);
        }
    }

    energy
}

// E = c * (k - 1)
// dE/dr1_i = -3 * c * (2k - 1) * v^-1 * r_i
// c = 4 * e * k
// k = u^3
// u = s^2 * v^-1
// v = r^2
#[inline(always)]
fn compute_pair_lj_forces_block<
    const D: usize, // Number of dimensions
    const M: usize, // Number of A particles
>(
    coords_a: [Simd<f32, M>; D],
    coords_b: &[[f32; D]],
    sigma: f32,
    eps: f32,
) -> (Simd<f32, M>, [[f32; M]; D])
where
    LaneCount<M>: SupportedLaneCount,
{
    let mut energy_acc = Simd::splat(0.0);

    let ra = coords_a.map(Simd::from);

    let sigma_vec = Simd::splat(sigma);
    let eps_vec = Simd::splat(eps);

    // let mut forces_b = [[0.0; N]; D];

    let mut forces_a = [Simd::splat(0.0); D];

    // Outer loop over B coords
    for rb in coords_b {
        let rb = rb.map(Simd::splat);

        let mut r = [Simd::splat(0.0); D];

        for i in 0..D {
            r[i] = ra[i] - rb[i];
        }

        let mut v = Simd::splat(0.0);

        for r in r {
            v = r.mul_add(r, v);
        }

        let vinv = Simd::splat(1.0) / v;

        let u = sigma_vec * sigma_vec * vinv;
        let k = u * u * u;

        let c = Simd::splat(4.0) * eps_vec * k;

        energy_acc = c.mul_add(k - Simd::splat(1.0), energy_acc);

        let grad_coeff =
            Simd::splat(-3.0) * c * (k + k - Simd::splat(1.0)) * vinv;

        for d in 0..D {
            forces_a[d] -= grad_coeff * r[d];
        }
    }

    let forces_a = forces_a.map(Simd::into);

    (energy_acc, forces_a)
}

pub fn compute_pair_lj_forces_blockwise<const D: usize>(
    coords_a: &[[f32; D]],
    coords_b: &[[f32; D]],
    forces_a: &mut [[f32; D]],
    sigma: f32,
    eps: f32,
) -> f32 {
    let (a_chunks, a_rest) = coords_a.as_chunks::<16>();
    let (f_chunks, f_rest) = forces_a.as_chunks_mut();

    let mut energy_vec = Simd::splat(0.0);

    for (coords, forces) in a_chunks.iter().zip(f_chunks) {
        let coords_t = transpose_nd(*coords).map(Simd::from);
        let (energy_contrib, forces_t) =
            compute_pair_lj_forces_block(coords_t, coords_b, sigma, eps);

        energy_vec += energy_contrib;

        *forces = untranspose_nd(forces_t);
    }

    let energy = energy_vec.reduce_sum();
    let mut energy_vec = Simd::splat(0.0);

    let (a_chunks, a_rest) = a_rest.as_chunks::<8>();
    let (f_chunks, f_rest) = f_rest.as_chunks_mut();

    for (coords, forces) in a_chunks.iter().zip(f_chunks) {
        let coords_t = transpose_nd(*coords).map(Simd::from);
        let (energy_contrib, forces_t) =
            compute_pair_lj_forces_block(coords_t, coords_b, sigma, eps);

        energy_vec += energy_contrib;

        *forces = untranspose_nd(forces_t);
    }

    if a_rest.is_empty() {
        return energy;
    }

    let mut a_pad = [[0.0; D]; 8];
    for (dest, &src) in a_pad.iter_mut().zip(a_rest) {
        *dest = src;
    }

    let coords_t = transpose_nd(a_pad);
    let (energy_contrib, forces_t) = compute_pair_lj_forces_block(
        coords_t.map(Simd::from),
        coords_b,
        sigma,
        eps,
    );

    energy_vec += energy_contrib;

    let energy = energy + energy_vec.reduce_sum();

    let f_pad = untranspose_nd(forces_t);
    for (dest, &src) in f_rest.iter_mut().zip(&f_pad) {
        *dest = src;
    }

    energy
}
