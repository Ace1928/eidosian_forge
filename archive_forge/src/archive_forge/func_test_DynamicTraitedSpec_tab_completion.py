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
def test_DynamicTraitedSpec_tab_completion():

    def extract_func(list_out):
        return list_out[0]
    func_interface = Function(input_names=['list_out'], output_names=['out_file', 'another_file'], function=extract_func)
    list_extract = Node(Function(input_names=['list_out'], output_names=['out_file'], function=extract_func), name='list_extract')
    expected_input = set(list_extract.inputs.editable_traits())
    assert set(func_interface.inputs.__all__) == expected_input
    assert set(list_extract.inputs.__all__) == expected_input
    expected_output = set(list_extract.outputs.editable_traits())
    assert set(list_extract.outputs.__all__) == expected_output
    list_extract._interface._output_names.append('added_out_trait')
    expected_output.add('added_out_trait')
    assert set(list_extract.outputs.__all__) == expected_output