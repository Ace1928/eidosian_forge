import os
import pytest
import rpy2.robjects as robjects
import rpy2.robjects.help as rh
def test_iteritems(self):
    base_help = rh.Package('base')
    p = base_help.fetch('print')
    res = tuple(p.items())
    assert len(res) > 0