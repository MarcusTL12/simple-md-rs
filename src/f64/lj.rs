use std::simd::{LaneCount, StdFloat, SupportedLaneCount, prelude::*};

fn array_sum<const N: usize>(x: [f64; N], y: [f64; N]) -> [f64; N] {
    let mut ans = [0.0; N];

    for i in 0..N {
        ans[i] = x[i] + y[i];
    }

    ans
}

fn array_diff<const N: usize>(x: [f64; N], y: [f64; N]) -> [f64; N] {
    let mut ans = [0.0; N];

    for i in 0..N {
        ans[i] = x[i] - y[i];
    }

    ans
}

// Returns sum(u_i * v_i)
fn dot<const N: usize>(u: [f64; N], v: [f64; N]) -> f64 {
    let mut ans = 0.0;

    for (a, b) in u.into_iter().zip(v) {
        ans = a.mul_add(b, ans);
    }

    ans
}

// E = 4 * e * (k^2 - k) = 4 * e * k * (k - 1)
// k = (s/r)^6
// r = |r1 - r2|
pub fn lj_energy<const N: usize>(
    r1: [f64; N],
    r2: [f64; N],
    sigma: f64,
    eps: f64,
) -> f64 {
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
fn lj_gradient<const N: usize>(
    r1: [f64; N],
    r2: [f64; N],
    sigma: f64,
    eps: f64,
) -> (f64, [f64; N]) {
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
    coords: &[[f64; N]],
    forces: &mut [[f64; N]],
    sigma: f64,
    eps: f64,
) -> f64 {
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
pub fn compute_pair_lj_forces_block<
    const D: usize, // Number of dimensions
    const M: usize, // Number of A particles
>(
    coords_a: [[f64; M]; D],
    coords_b: &[[f64; D]],
    sigma: f64,
    eps: f64,
) -> (f64, [[f64; M]; D])
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

    let energy = energy_acc.reduce_sum();
    let forces_a = forces_a.map(Simd::into);

    (energy, forces_a)
}

pub fn compute_lj_forces_blockwise<const N: usize>(
    coords: &[[f64; N]],
    forces: &mut [[f64; N]],
    sigma: f64,
    eps: f64,
) -> f64 {
    todo!()
}
