import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_input_version_6():

    class DerivedInterface1(nib.BaseInterface):
        input_spec = MaxVerInputSpec
        _version = '0.7'
    obj = DerivedInterface1()
    obj.inputs.foo = 1
    obj._check_version_requirements(obj.inputs)