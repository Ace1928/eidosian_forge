import unittest
@njit
def get_field_sum(rec):
    out = 0
    for f in literal_unroll(fields_gl):
        out += rec[f]
    return out