import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_bool():
    sexp = ri.vector([True], ri.RTYPES.LGLSXP)
    isLogical = ri.globalenv.find('is.logical')
    assert isLogical(sexp)[0]
    assert sexp[0] is True
    sexp = ri.vector(['a'], ri.RTYPES.LGLSXP)
    isLogical = ri.globalenv.find('is.logical')
    assert isLogical(sexp)[0]
    assert sexp[0]