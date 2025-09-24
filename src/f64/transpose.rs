use std::{
    mem::transmute,
    simd::{LaneCount, SupportedLaneCount, prelude::*},
};

#[inline(always)]
fn transpose_2d<const M: usize>(rows: [[f64; 2]; M]) -> [[f64; M]; 2]
where
    LaneCount<M>: SupportedLaneCount,
{
    let vecs: &[[f64; M]] =
        unsafe { rows.as_flattened().as_chunks_unchecked() };

    let v0 = Simd::from(vecs[0]);
    let v1 = Simd::from(vecs[1]);

    let (u0, u1) = v0.deinterleave(v1);

    [u0.into(), u1.into()]
}

#[inline(always)]
fn transpose_3d_2(rows: [[f64; 3]; 2]) -> [[f64; 2]; 3] {
    let mut data = [0.0; 6];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    // [[0, 2, 4],
    //  [1, 3, 5]]

    let v = simd_swizzle!(Simd::from(data), [0, 2, 4, 1, 3, 5]);

    let data: [f64; 6] = v.into();
    let (c, _) = data.as_chunks();

    [c[0], c[1], c[2]]
}

#[inline(always)]
fn transpose_3d_4(rows: [[f64; 3]; 4]) -> [[f64; 4]; 3] {
    let mut data = [0.0; 12];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    // [[0, 4,  8],
    //  [1, 5,  9],
    //  [2, 6, 10],
    //  [3, 7, 11]]

    let v =
        simd_swizzle!(Simd::from(data), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);

    let data: [f64; 12] = v.into();
    let (c, _) = data.as_chunks();

    [c[0], c[1], c[2]]
}

#[inline(always)]
fn transpose_3d_8(rows: [[f64; 3]; 8]) -> [[f64; 8]; 3] {
    let mut data = [0.0; 24];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    // [[0,  8, 16],
    //  [1,  9, 17],
    //  [2, 10, 18],
    //  [3, 11, 19],
    //  [4, 12, 20],
    //  [5, 13, 21],
    //  [6, 14, 22],
    //  [7, 15, 23]]

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13, 21, 6,
            14, 22, 7, 15, 23
        ]
    );

    let data: [f64; 24] = v.into();
    let (c, _) = data.as_chunks();

    [c[0], c[1], c[2]]
}

#[inline(always)]
fn transpose_3d_16(rows: [[f64; 3]; 16]) -> [[f64; 16]; 3] {
    let mut data = [0.0; 48];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    // [[ 0, 16, 32],
    //  [ 1, 17, 33],
    //  [ 2, 18, 34],
    //  [ 3, 19, 35],
    //  [ 4, 20, 36],
    //  [ 5, 21, 37],
    //  [ 6, 22, 38],
    //  [ 7, 23, 39],
    //  [ 8, 24, 40],
    //  [ 9, 25, 41],
    //  [10, 26, 42],
    //  [11, 27, 43],
    //  [12, 28, 44],
    //  [13, 29, 45],
    //  [14, 30, 46],
    //  [15, 31, 47]]

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37,
            6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43,
            12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47
        ]
    );

    let data: [f64; 48] = v.into();
    let (c, _) = data.as_chunks();

    [c[0], c[1], c[2]]
}

pub fn transpose_nd<const D: usize, const M: usize>(
    rows: [[f64; D]; M],
) -> [[f64; M]; D]
where
    LaneCount<M>: SupportedLaneCount,
{
    match D {
        2 => {
            let rows: &[[f64; 2]; M] = unsafe { transmute(&rows) };
            let cols = transpose_2d(*rows);
            let cols: &[[f64; M]; D] = unsafe { transmute(&cols) };
            return *cols;
        }
        3 => match M {
            2 => {
                let rows: &[[f64; 3]; 2] = unsafe { transmute(&rows) };
                let cols = transpose_3d_2(*rows);
                let cols: &[[f64; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            4 => {
                let rows: &[[f64; 3]; 4] = unsafe { transmute(&rows) };
                let cols = transpose_3d_4(*rows);
                let cols: &[[f64; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            8 => {
                let rows: &[[f64; 3]; 8] = unsafe { transmute(&rows) };
                let cols = transpose_3d_8(*rows);
                let cols: &[[f64; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            16 => {
                let rows: &[[f64; 3]; 16] = unsafe { transmute(&rows) };
                let cols = transpose_3d_16(*rows);
                let cols: &[[f64; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            _ => (),
        },
        _ => (),
    }

    let mut cols = [[0.0; M]; D];

    for i in 0..D {
        for j in 0..M {
            cols[i][j] = rows[j][i];
        }
    }

    cols
}
