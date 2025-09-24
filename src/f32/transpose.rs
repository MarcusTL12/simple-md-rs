use std::{
    mem::transmute,
    simd::{LaneCount, SupportedLaneCount, prelude::*},
};

#[inline(always)]
fn transpose_2d<const M: usize>(rows: [[f32; 2]; M]) -> [[f32; M]; 2]
where
    LaneCount<M>: SupportedLaneCount,
{
    let vecs: &[[f32; M]] =
        unsafe { rows.as_flattened().as_chunks_unchecked() };

    let v0 = Simd::from(vecs[0]);
    let v1 = Simd::from(vecs[1]);

    let (u0, u1) = v0.deinterleave(v1);

    [u0.into(), u1.into()]
}

#[inline(always)]
fn transpose_3d_2(rows: [[f32; 3]; 2]) -> [[f32; 2]; 3] {
    let mut data = [0.0; 6];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(Simd::from(data), [0, 3, 1, 4, 2, 5]);

    let data: [f32; 6] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn transpose_3d_4(rows: [[f32; 3]; 4]) -> [[f32; 4]; 3] {
    let mut data = [0.0; 12];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v =
        simd_swizzle!(Simd::from(data), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);

    let data: [f32; 12] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn transpose_3d_8(rows: [[f32; 3]; 8]) -> [[f32; 8]; 3] {
    let mut data = [0.0; 24];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22, 2, 5, 8,
            11, 14, 17, 20, 23
        ]
    );

    let data: [f32; 24] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn transpose_3d_16(rows: [[f32; 3]; 16]) -> [[f32; 16]; 3] {
    let mut data = [0.0; 48];
    for (i, &x) in rows.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 1, 4,
            7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 2, 5, 8, 11,
            14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47
        ]
    );

    let data: [f32; 48] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

pub fn transpose_nd<const D: usize, const M: usize>(
    rows: [[f32; D]; M],
) -> [[f32; M]; D]
where
    LaneCount<M>: SupportedLaneCount,
{
    match D {
        2 => {
            let rows: &[[f32; 2]; M] = unsafe { transmute(&rows) };
            let cols = transpose_2d(*rows);
            let cols: &[[f32; M]; D] = unsafe { transmute(&cols) };
            return *cols;
        }
        3 => match M {
            2 => {
                let rows: &[[f32; 3]; 2] = unsafe { transmute(&rows) };
                let cols = transpose_3d_2(*rows);
                let cols: &[[f32; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            4 => {
                let rows: &[[f32; 3]; 4] = unsafe { transmute(&rows) };
                let cols = transpose_3d_4(*rows);
                let cols: &[[f32; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            8 => {
                let rows: &[[f32; 3]; 8] = unsafe { transmute(&rows) };
                let cols = transpose_3d_8(*rows);
                let cols: &[[f32; M]; D] = unsafe { transmute(&cols) };
                return *cols;
            }
            16 => {
                let rows: &[[f32; 3]; 16] = unsafe { transmute(&rows) };
                let cols = transpose_3d_16(*rows);
                let cols: &[[f32; M]; D] = unsafe { transmute(&cols) };
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

#[inline(always)]
fn untranspose_2d<const M: usize>(rows: [[f32; M]; 2]) -> [[f32; 2]; M]
where
    LaneCount<M>: SupportedLaneCount,
{
    let v0 = Simd::from(rows[0]);
    let v1 = Simd::from(rows[1]);

    let (u0, u1) = v0.interleave(v1);

    unsafe {
        [u0.into(), u1.into()]
            .as_flattened()
            .as_chunks_unchecked()
            .as_chunks_unchecked()[0]
    }
}

#[inline(always)]
fn untranspose_3d_2(cols: [[f32; 2]; 3]) -> [[f32; 3]; 2] {
    let mut data = [0.0; 6];
    for (i, &x) in cols.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(Simd::from(data), [0, 2, 4, 1, 3, 5]);

    let data: [f32; 6] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn untranspose_3d_4(cols: [[f32; 4]; 3]) -> [[f32; 3]; 4] {
    let mut data = [0.0; 12];
    for (i, &x) in cols.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v =
        simd_swizzle!(Simd::from(data), [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);

    let data: [f32; 12] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn untranspose_3d_8(cols: [[f32; 8]; 3]) -> [[f32; 3]; 8] {
    let mut data = [0.0; 24];
    for (i, &x) in cols.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13, 21, 6,
            14, 22, 7, 15, 23
        ]
    );

    let data: [f32; 24] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

#[inline(always)]
fn untranspose_3d_16(cols: [[f32; 16]; 3]) -> [[f32; 3]; 16] {
    let mut data = [0.0; 48];
    for (i, &x) in cols.as_flattened().iter().enumerate() {
        data[i] = x;
    }

    let v = simd_swizzle!(
        Simd::from(data),
        [
            0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37,
            6, 22, 38, 7, 23, 39, 8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43,
            12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47
        ]
    );

    let data: [f32; 48] = v.into();
    unsafe { data.as_chunks_unchecked().as_chunks_unchecked()[0] }
}

pub fn untranspose_nd<const D: usize, const M: usize>(
    cols: [[f32; M]; D],
) -> [[f32; D]; M]
where
    LaneCount<M>: SupportedLaneCount,
{
    match D {
        2 => {
            let cols: &[[f32; M]; 2] = unsafe { transmute(&cols) };
            let rows = untranspose_2d(*cols);
            let rows: &[[f32; D]; M] = unsafe { transmute(&rows) };
            return *rows;
        }
        3 => match M {
            2 => {
                let cols: &[[f32; 2]; 3] = unsafe { transmute(&cols) };
                let rows = untranspose_3d_2(*cols);
                let rows: &[[f32; D]; M] = unsafe { transmute(&rows) };
                return *rows;
            }
            4 => {
                let cols: &[[f32; 4]; 3] = unsafe { transmute(&cols) };
                let rows = untranspose_3d_4(*cols);
                let rows: &[[f32; D]; M] = unsafe { transmute(&rows) };
                return *rows;
            }
            8 => {
                let cols: &[[f32; 8]; 3] = unsafe { transmute(&cols) };
                let rows = untranspose_3d_8(*cols);
                let rows: &[[f32; D]; M] = unsafe { transmute(&rows) };
                return *rows;
            }
            16 => {
                let cols: &[[f32; 16]; 3] = unsafe { transmute(&cols) };
                let rows = untranspose_3d_16(*cols);
                let rows: &[[f32; D]; M] = unsafe { transmute(&rows) };
                return *rows;
            }
            _ => (),
        },
        _ => (),
    }

    let mut rows = [[0.0; D]; M];

    for i in 0..M {
        for j in 0..D {
            rows[i][j] = cols[j][i];
        }
    }

    rows
}
