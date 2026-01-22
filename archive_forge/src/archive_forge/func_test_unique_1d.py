import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_1d(self):

    def check_all(a, b, i1, i2, c, dt):
        base_msg = 'check {0} failed for type {1}'
        msg = base_msg.format('values', dt)
        v = unique(a)
        assert_array_equal(v, b, msg)
        msg = base_msg.format('return_index', dt)
        v, j = unique(a, True, False, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, i1, msg)
        msg = base_msg.format('return_inverse', dt)
        v, j = unique(a, False, True, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, i2, msg)
        msg = base_msg.format('return_counts', dt)
        v, j = unique(a, False, False, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j, c, msg)
        msg = base_msg.format('return_index and return_inverse', dt)
        v, j1, j2 = unique(a, True, True, False)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, i2, msg)
        msg = base_msg.format('return_index and return_counts', dt)
        v, j1, j2 = unique(a, True, False, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, c, msg)
        msg = base_msg.format('return_inverse and return_counts', dt)
        v, j1, j2 = unique(a, False, True, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i2, msg)
        assert_array_equal(j2, c, msg)
        msg = base_msg.format('return_index, return_inverse and return_counts', dt)
        v, j1, j2, j3 = unique(a, True, True, True)
        assert_array_equal(v, b, msg)
        assert_array_equal(j1, i1, msg)
        assert_array_equal(j2, i2, msg)
        assert_array_equal(j3, c, msg)
    a = [5, 7, 1, 2, 1, 5, 7] * 10
    b = [1, 2, 5, 7]
    i1 = [2, 3, 0, 1]
    i2 = [2, 3, 0, 1, 0, 2, 3] * 10
    c = np.multiply([2, 1, 2, 2], 10)
    types = []
    types.extend(np.typecodes['AllInteger'])
    types.extend(np.typecodes['AllFloat'])
    types.append('datetime64[D]')
    types.append('timedelta64[D]')
    for dt in types:
        aa = np.array(a, dt)
        bb = np.array(b, dt)
        check_all(aa, bb, i1, i2, c, dt)
    dt = 'O'
    aa = np.empty(len(a), dt)
    aa[:] = a
    bb = np.empty(len(b), dt)
    bb[:] = b
    check_all(aa, bb, i1, i2, c, dt)
    dt = [('', 'i'), ('', 'i')]
    aa = np.array(list(zip(a, a)), dt)
    bb = np.array(list(zip(b, b)), dt)
    check_all(aa, bb, i1, i2, c, dt)
    aa = [1.0 + 0j, 1 - 1j, 1]
    assert_array_equal(np.unique(aa), [1.0 - 1j, 1.0 + 0j])
    a = [(1, 2), (1, 2), (2, 3)]
    unq = [1, 2, 3]
    inv = [0, 1, 0, 1, 1, 2]
    a1 = unique(a)
    assert_array_equal(a1, unq)
    a2, a2_inv = unique(a, return_inverse=True)
    assert_array_equal(a2, unq)
    assert_array_equal(a2_inv, inv)
    a = np.chararray(5)
    a[...] = ''
    a2, a2_inv = np.unique(a, return_inverse=True)
    assert_array_equal(a2_inv, np.zeros(5))
    a = []
    a1_idx = np.unique(a, return_index=True)[1]
    a2_inv = np.unique(a, return_inverse=True)[1]
    a3_idx, a3_inv = np.unique(a, return_index=True, return_inverse=True)[1:]
    assert_equal(a1_idx.dtype, np.intp)
    assert_equal(a2_inv.dtype, np.intp)
    assert_equal(a3_idx.dtype, np.intp)
    assert_equal(a3_inv.dtype, np.intp)
    a = [2.0, np.nan, 1.0, np.nan]
    ua = [1.0, 2.0, np.nan]
    ua_idx = [2, 0, 1]
    ua_inv = [1, 2, 0, 2]
    ua_cnt = [1, 1, 2]
    assert_equal(np.unique(a), ua)
    assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
    assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
    assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))
    a = [2.0 - 1j, np.nan, 1.0 + 1j, complex(0.0, np.nan), complex(1.0, np.nan)]
    ua = [1.0 + 1j, 2.0 - 1j, complex(0.0, np.nan)]
    ua_idx = [2, 0, 3]
    ua_inv = [1, 2, 0, 2, 2]
    ua_cnt = [1, 1, 3]
    assert_equal(np.unique(a), ua)
    assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
    assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
    assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))
    nat = np.datetime64('nat')
    a = [np.datetime64('2020-12-26'), nat, np.datetime64('2020-12-24'), nat]
    ua = [np.datetime64('2020-12-24'), np.datetime64('2020-12-26'), nat]
    ua_idx = [2, 0, 1]
    ua_inv = [1, 2, 0, 2]
    ua_cnt = [1, 1, 2]
    assert_equal(np.unique(a), ua)
    assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
    assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
    assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))
    nat = np.timedelta64('nat')
    a = [np.timedelta64(1, 'D'), nat, np.timedelta64(1, 'h'), nat]
    ua = [np.timedelta64(1, 'h'), np.timedelta64(1, 'D'), nat]
    ua_idx = [2, 0, 1]
    ua_inv = [1, 2, 0, 2]
    ua_cnt = [1, 1, 2]
    assert_equal(np.unique(a), ua)
    assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
    assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
    assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))
    all_nans = [np.nan] * 4
    ua = [np.nan]
    ua_idx = [0]
    ua_inv = [0, 0, 0, 0]
    ua_cnt = [4]
    assert_equal(np.unique(all_nans), ua)
    assert_equal(np.unique(all_nans, return_index=True), (ua, ua_idx))
    assert_equal(np.unique(all_nans, return_inverse=True), (ua, ua_inv))
    assert_equal(np.unique(all_nans, return_counts=True), (ua, ua_cnt))