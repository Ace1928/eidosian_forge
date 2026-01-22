import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_extract_silent_index_error():
    seq_R = robjects.baseenv['seq']
    mySeq = seq_R(0, 10)
    myIndex = robjects.vectors.StrVector(['a', 'b', 'c'])
    with utils.obj_in_module(rpy2.rinterface_lib.callbacks, 'consolewrite_print', utils.noconsole):
        res = mySeq.rx(myIndex)
        assert all((x == rpy2.robjects.NA_Integer for x in res))