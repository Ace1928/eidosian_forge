import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_input_version_5():

    class DerivedInterface2(nib.BaseInterface):
        input_spec = MaxVerInputSpec
        _version = '0.8'
    obj = DerivedInterface2()
    obj.inputs.foo = 1
    with pytest.raises(Exception) as excinfo:
        obj._check_version_requirements(obj.inputs)
    assert 'version 0.8 > required 0.7' in str(excinfo.value)