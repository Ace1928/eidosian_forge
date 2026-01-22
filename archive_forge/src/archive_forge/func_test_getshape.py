import rpy2.rinterface as rinterface
from rpy2.rinterface import bufferprotocol
def test_getshape():
    v = rinterface.IntSexpVector([1, 2, 3])
    assert bufferprotocol.getshape(v.__sexp__._cdata, 1) == (3,)
    m = rinterface.baseenv.find('matrix')(nrow=2, ncol=3)
    assert bufferprotocol.getshape(m.__sexp__._cdata, 2) == (2, 3)