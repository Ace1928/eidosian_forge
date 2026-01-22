import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def check_promotion_cases(self, promote_func):
    b = np.bool_(0)
    i8, i16, i32, i64 = (np.int8(0), np.int16(0), np.int32(0), np.int64(0))
    u8, u16, u32, u64 = (np.uint8(0), np.uint16(0), np.uint32(0), np.uint64(0))
    f32, f64, fld = (np.float32(0), np.float64(0), np.longdouble(0))
    c64, c128, cld = (np.complex64(0), np.complex128(0), np.clongdouble(0))
    assert_equal(promote_func(i8, i16), np.dtype(np.int16))
    assert_equal(promote_func(i32, i8), np.dtype(np.int32))
    assert_equal(promote_func(i16, i64), np.dtype(np.int64))
    assert_equal(promote_func(u8, u32), np.dtype(np.uint32))
    assert_equal(promote_func(f32, f64), np.dtype(np.float64))
    assert_equal(promote_func(fld, f32), np.dtype(np.longdouble))
    assert_equal(promote_func(f64, fld), np.dtype(np.longdouble))
    assert_equal(promote_func(c128, c64), np.dtype(np.complex128))
    assert_equal(promote_func(cld, c128), np.dtype(np.clongdouble))
    assert_equal(promote_func(c64, fld), np.dtype(np.clongdouble))
    assert_equal(promote_func(b, i32), np.dtype(np.int32))
    assert_equal(promote_func(b, u8), np.dtype(np.uint8))
    assert_equal(promote_func(i8, u8), np.dtype(np.int16))
    assert_equal(promote_func(u8, i32), np.dtype(np.int32))
    assert_equal(promote_func(i64, u32), np.dtype(np.int64))
    assert_equal(promote_func(u64, i32), np.dtype(np.float64))
    assert_equal(promote_func(i32, f32), np.dtype(np.float64))
    assert_equal(promote_func(i64, f32), np.dtype(np.float64))
    assert_equal(promote_func(f32, i16), np.dtype(np.float32))
    assert_equal(promote_func(f32, u32), np.dtype(np.float64))
    assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
    assert_equal(promote_func(c128, f32), np.dtype(np.complex128))
    assert_equal(promote_func(cld, f64), np.dtype(np.clongdouble))
    assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
    assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
    assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
    assert_equal(promote_func(np.array([b]), u32), np.dtype(np.uint32))
    assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
    assert_equal(promote_func(u64, np.array([i32])), np.dtype(np.int32))
    assert_equal(promote_func(i64, np.array([u32])), np.dtype(np.uint32))
    assert_equal(promote_func(np.int32(-1), np.array([u64])), np.dtype(np.float64))
    assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))
    assert_equal(promote_func(fld, np.array([f32])), np.dtype(np.float32))
    assert_equal(promote_func(np.array([f64]), fld), np.dtype(np.float64))
    assert_equal(promote_func(fld, np.array([c64])), np.dtype(np.complex64))
    assert_equal(promote_func(c64, np.array([f64])), np.dtype(np.complex128))
    assert_equal(promote_func(np.complex64(3j), np.array([f64])), np.dtype(np.complex128))
    assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
    assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
    assert_equal(promote_func(np.array([b]), u64), np.dtype(np.uint64))
    assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))
    assert_equal(promote_func(np.array([u16]), f64), np.dtype(np.float64))
    assert_equal(promote_func(np.array([u16]), i32), np.dtype(np.uint16))
    assert_equal(promote_func(np.array([f32]), c128), np.dtype(np.complex64))