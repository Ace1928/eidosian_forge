import pytest
import contextlib
import os
import tempfile
from rpy2.robjects.packages import importr, data
from rpy2.robjects import r
from rpy2.robjects.lib import grdevices
@pytest.mark.xfail(os.name == 'nt', reason='Windows produces non-empty file with no plot')
def test_rendertobytes_noplot():
    with grdevices.render_to_bytesio(grdevices.png) as b:
        pass
    assert len(b.getvalue()) == 0