import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.adapters import output_adapters
def test_unsupported_types_error():
    adapter = output_adapters.ClassificationAdapter(name='a')
    with pytest.raises(TypeError) as info:
        adapter.adapt(1, batch_size=32)
    assert 'Expect the target data of a to be tf' in str(info.value)