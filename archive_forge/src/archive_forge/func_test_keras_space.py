from tests.tune_tensorflow.mock import MockSpec
from tune_tensorflow.utils import (
def test_keras_space():
    space = keras_space(MockSpec, a=1, b=2)
    spec = extract_keras_spec(list(space)[0], {})
    assert spec == MockSpec
    spec = extract_keras_spec(list(space)[0], {to_keras_spec_expr(MockSpec): 'dummy'})
    assert 'dummy' == spec