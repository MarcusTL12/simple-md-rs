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
fn lj_energy<const N: usize>(
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
fn lj_gradient<const N: usize>(
    r1: [f64; N],
    r2: [f64; N],
    sigma: f64,
    eps: f64,
) -> [f64; N] {
    let r = array_diff(r1, r2);
    let v = dot(r, r);
    let vinv = 1.0 / v;

    let u = sigma * sigma * vinv;
    let k = u.powi(3);

    r.map(|ri| -12.0 * eps * (2.0 * k - 1.0) * k * vinv * ri)
}

fn main() {
    println!("Hello, world!");
}
