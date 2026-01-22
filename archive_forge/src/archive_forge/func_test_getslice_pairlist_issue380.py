import rpy2.rinterface as ri
def test_getslice_pairlist_issue380():
    vec = ri.baseenv['.Options']
    vec_slice = vec[0:2]
    assert len(vec_slice) == 2
    assert vec_slice.typeof == ri.RTYPES.LISTSXP
    assert vec.names[0] == vec_slice.names[0]
    assert vec.names[1] == vec_slice.names[1]