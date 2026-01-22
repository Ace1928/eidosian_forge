import rpy2.rinterface as ri
def test_getitem_pairlist():
    pairlist = ri.baseenv.find('pairlist')
    pl = pairlist(a=ri.StrSexpVector(['1']), b=ri.StrSexpVector(['3']))
    y = pl[0]
    assert y.typeof == ri.RTYPES.VECSXP
    assert len(y) == 1
    assert y[0][0] == '1'
    assert y.names[0] == 'a'