import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
def check_transform_produces_correct_output_type_forward(model, inputs, checker):
    outputs = model.predict(inputs)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=True)
    assert checker(inputs, outputs)
    outputs, _ = model(inputs, is_train=False)
    assert checker(inputs, outputs)