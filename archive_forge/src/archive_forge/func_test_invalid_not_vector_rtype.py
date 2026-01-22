import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_invalid_not_vector_rtype():
    with pytest.raises(ValueError):
        ri.vector([1], ri.RTYPES.ENVSXP)