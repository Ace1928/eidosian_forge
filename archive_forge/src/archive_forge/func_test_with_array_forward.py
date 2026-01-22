import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def test_with_array_forward(ragged_input, padded_input, list_input, array_input):
    for inputs in (ragged_input, padded_input, list_input, array_input):
        checker = get_data_checker(inputs)
        model = get_array_model()
        check_transform_produces_correct_output_type_forward(model, inputs, checker)