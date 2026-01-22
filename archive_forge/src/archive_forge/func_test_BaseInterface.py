import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_BaseInterface():
    config.set('monitoring', 'enable', '0')
    assert nib.BaseInterface.help() is None

    class InputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')
        goo = nib.traits.Int(desc='a random int', mandatory=True)
        moo = nib.traits.Int(desc='a random int', mandatory=False)
        hoo = nib.traits.Int(desc='a random int', usedefault=True)
        zoo = nib.File(desc='a file', copyfile=False)
        woo = nib.File(desc='a file', copyfile=True)

    class OutputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')

    class DerivedInterface(nib.BaseInterface):
        input_spec = InputSpec
        resource_monitor = False
    assert DerivedInterface.help() is None
    assert 'moo' in ''.join(_inputs_help(DerivedInterface))
    assert DerivedInterface()._outputs() is None
    assert DerivedInterface().inputs.foo == nib.Undefined
    with pytest.raises(ValueError):
        DerivedInterface()._check_mandatory_inputs()
    assert DerivedInterface(goo=1)._check_mandatory_inputs() is None
    with pytest.raises(ValueError):
        DerivedInterface().run()
    with pytest.raises(NotImplementedError):
        DerivedInterface(goo=1).run()

    class DerivedInterface2(DerivedInterface):
        output_spec = OutputSpec

        def _run_interface(self, runtime):
            return runtime
    assert DerivedInterface2.help() is None
    assert DerivedInterface2()._outputs().foo == nib.Undefined
    with pytest.raises(NotImplementedError):
        DerivedInterface2(goo=1).run()
    default_inpu_spec = nib.BaseInterface.input_spec
    nib.BaseInterface.input_spec = None
    with pytest.raises(Exception):
        nib.BaseInterface()
    nib.BaseInterface.input_spec = default_inpu_spec