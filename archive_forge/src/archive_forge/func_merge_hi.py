import collections
from numba.core import types
@wrap
def merge_hi(ms, keys, values, ssa, na, ssb, nb):
    """
        Merge the na elements starting at ssa with the nb elements starting at
        ssb = ssa + na in a stable way, in-place.  na and nb must be > 0,
        and should have na >= nb.  See listsort.txt for more info.

        An updated MergeState is returned (with possibly a different min_gallop
        or larger temp arrays).

        NOTE: compared to CPython's timsort, the requirement that
            "Must also have that keys[ssa + na - 1] belongs at the end of the merge"

        is removed. This makes the code a bit simpler and easier to reason about.
        """
    assert na > 0 and nb > 0 and (na >= nb)
    assert ssb == ssa + na
    ms = merge_getmem(ms, nb)
    sortslice_copy(ms.keys, ms.values, 0, keys, values, ssb, nb)
    a_keys = keys
    a_values = values
    b_keys = ms.keys
    b_values = ms.values
    dest = ssb + nb - 1
    ssb = nb - 1
    ssa = ssa + na - 1
    _has_values = has_values(b_keys, b_values)
    min_gallop = ms.min_gallop
    while nb > 0 and na > 0:
        acount = 0
        bcount = 0
        while True:
            if LT(b_keys[ssb], a_keys[ssa]):
                keys[dest] = a_keys[ssa]
                if _has_values:
                    values[dest] = a_values[ssa]
                dest -= 1
                ssa -= 1
                na -= 1
                if na == 0:
                    break
                acount += 1
                bcount = 0
                if acount >= min_gallop:
                    break
            else:
                keys[dest] = b_keys[ssb]
                if _has_values:
                    values[dest] = b_values[ssb]
                dest -= 1
                ssb -= 1
                nb -= 1
                if nb == 0:
                    break
                bcount += 1
                acount = 0
                if bcount >= min_gallop:
                    break
        if DO_GALLOP and na > 0 and (nb > 0):
            min_gallop += 1
            while acount >= MIN_GALLOP or bcount >= MIN_GALLOP:
                min_gallop -= min_gallop > 1
                k = gallop_right(b_keys[ssb], a_keys, ssa - na + 1, ssa + 1, ssa)
                k = ssa + 1 - k
                acount = k
                if k > 0:
                    sortslice_copy_down(keys, values, dest, a_keys, a_values, ssa, k)
                    dest -= k
                    ssa -= k
                    na -= k
                    if na == 0:
                        break
                keys[dest] = b_keys[ssb]
                if _has_values:
                    values[dest] = b_values[ssb]
                dest -= 1
                ssb -= 1
                nb -= 1
                if nb == 0:
                    break
                k = gallop_left(a_keys[ssa], b_keys, ssb - nb + 1, ssb + 1, ssb)
                k = ssb + 1 - k
                bcount = k
                if k > 0:
                    sortslice_copy_down(keys, values, dest, b_keys, b_values, ssb, k)
                    dest -= k
                    ssb -= k
                    nb -= k
                    if nb == 0:
                        break
                keys[dest] = a_keys[ssa]
                if _has_values:
                    values[dest] = a_values[ssa]
                dest -= 1
                ssa -= 1
                na -= 1
                if na == 0:
                    break
            min_gallop += 1
    if na == 0:
        sortslice_copy(keys, values, dest - nb + 1, b_keys, b_values, ssb - nb + 1, nb)
    else:
        assert nb == 0
        assert dest == ssa
    return merge_adjust_gallop(ms, min_gallop)