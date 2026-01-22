from snappy.pari import pari
def is_pari_col_vector(obj):
    return isinstance(obj, PariGen) and obj.type() == 't_COL'