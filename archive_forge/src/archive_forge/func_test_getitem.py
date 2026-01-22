import array
import pytest
import rpy2.robjects as robjects
def test_getitem():
    letters_R = robjects.r['letters']
    assert isinstance(letters_R, robjects.Vector)
    letters = (('a', 0), ('b', 1), ('c', 2), ('x', 23), ('y', 24), ('z', 25))
    for l, i in letters:
        assert letters_R[i] == l
    as_list_R = robjects.r['as.list']
    seq_R = robjects.r['seq']
    mySeq = seq_R(0, 10)
    myList = as_list_R(mySeq)
    for i, li in enumerate(myList):
        assert myList[i][0] == i