import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_input_version_1():

    class DerivedInterface1(nib.BaseInterface):
        input_spec = MinVerInputSpec
    obj = DerivedInterface1()
    obj._check_version_requirements(obj.inputs)
    config.set('execution', 'stop_on_unknown_version', True)
    with pytest.raises(ValueError) as excinfo:
        obj._check_version_requirements(obj.inputs)
    assert 'no version information' in str(excinfo.value)
    config.set_default_config()