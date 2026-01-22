import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
def test_config_reserved_aliases():
    """Test that the auto-generated pydantic schemas auto-alias reserved
    attributes like "validate" that would otherwise cause NameError."""

    @my_registry.cats('catsie.with_alias')
    def catsie_with_alias(validate: StrictBool=False):
        return validate
    cfg = {'@cats': 'catsie.with_alias', 'validate': True}
    resolved = my_registry.resolve({'test': cfg})
    filled = my_registry.fill({'test': cfg})
    assert resolved['test'] is True
    assert filled['test'] == cfg
    cfg = {'@cats': 'catsie.with_alias', 'validate': 20}
    with pytest.raises(ConfigValidationError):
        my_registry.resolve({'test': cfg})