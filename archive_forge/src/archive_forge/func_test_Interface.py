import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_Interface():
    assert nib.Interface.input_spec is None
    assert nib.Interface.output_spec is None
    with pytest.raises(NotImplementedError):
        nib.Interface()

    class DerivedInterface(nib.Interface):

        def __init__(self):
            pass
    nif = DerivedInterface()
    with pytest.raises(NotImplementedError):
        nif.run()
    with pytest.raises(NotImplementedError):
        nif.aggregate_outputs()
    with pytest.raises(NotImplementedError):
        nif._list_outputs()