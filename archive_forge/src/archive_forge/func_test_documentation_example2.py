import unittest
def test_documentation_example2(self):
    import numpy as np
    from numba import njit, literal_unroll
    arr = np.array([(1, 2)], dtype=[('a1', 'f8'), ('a2', 'f8')])
    fields_gl = ('a1', 'a2')

    @njit
    def get_field_sum(rec):
        out = 0
        for f in literal_unroll(fields_gl):
            out += rec[f]
        return out
    get_field_sum(arr[0])
    self.assertEqual(get_field_sum(arr[0]), 3)