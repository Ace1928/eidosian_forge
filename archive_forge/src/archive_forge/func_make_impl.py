import fractions
import pytest
import cirq
def make_impl(i, op):

    def impl(x, y):
        if isinstance(y, MockValue):
            return op(x.val, y.val)
        if bad_index == i:
            return bad_result
        return NotImplemented
    return impl