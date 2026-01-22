@staticmethod
def mismatch_region(next_a, region_ia, next_b, region_ib):
    if next_a < region_ia or next_b < region_ib:
        return ('conflict', None, None, next_a, region_ia, next_b, region_ib)