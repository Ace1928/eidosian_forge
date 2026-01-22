import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_missing_R_Preserve_object_bug():
    rgc = ri.baseenv['gc']
    xx = range(100000)
    x = ri.IntSexpVector(xx)
    rgc()
    assert x[0] == 0