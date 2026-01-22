import inspect
import pickle
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import catalogue
import numpy
import pytest
import thinc.config
from thinc.api import Config, Model, NumpyOps, RAdam
from thinc.config import ConfigValidationError
from thinc.types import Generator, Ragged
from thinc.util import partial
from .util import make_tempdir
def test_handle_generic_model_type():
    """Test that validation can handle checks against arbitrary generic
    types in function argument annotations."""

    @my_registry.layers('my_transform.v1')
    def my_transform(model: Model[int, int]):
        model.name = 'transformed_model'
        return model
    cfg = {'@layers': 'my_transform.v1', 'model': {'@layers': 'Linear.v1'}}
    model = my_registry.resolve({'test': cfg})['test']
    assert isinstance(model, Model)
    assert model.name == 'transformed_model'