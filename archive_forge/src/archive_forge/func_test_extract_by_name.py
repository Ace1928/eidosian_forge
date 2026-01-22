import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_extract_by_name():
    seq_R = robjects.baseenv['seq']
    mySeq = seq_R(0, 25)
    letters = robjects.baseenv['letters']
    mySeq = robjects.baseenv['names<-'](mySeq, letters)
    myIndex = robjects.vectors.StrVector(letters[2])
    mySubset = mySeq.rx(myIndex)
    for i, si in enumerate(myIndex):
        assert mySubset[i] == 2