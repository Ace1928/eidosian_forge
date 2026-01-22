import os
import warnings
import pytest
from ....utils.filemanip import split_filename
from ... import base as nib
from ...base import traits, Undefined
from ....interfaces import fsl
from ...utility.wrappers import Function
from ....pipeline import Node
from ..specs import get_filecopy_info
def test_TraitedSpec_tab_completion():
    bet_nd = Node(fsl.BET(), name='bet')
    bet_interface = fsl.BET()
    bet_inputs = bet_nd.inputs.class_editable_traits()
    bet_outputs = bet_nd.outputs.class_editable_traits()
    assert set(bet_nd.inputs.__all__) == set(bet_inputs)
    assert set(bet_interface.inputs.__all__) == set(bet_inputs)
    assert set(bet_nd.outputs.__all__) == set(bet_outputs)