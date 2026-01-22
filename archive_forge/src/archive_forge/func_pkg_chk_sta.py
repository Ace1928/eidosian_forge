import sys
import types
import pytest
from ..sexts import package_check
def pkg_chk_sta(*args, **kwargs):
    st_args = {}
    package_check(*args, setuptools_args=st_args, **kwargs)
    return st_args