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
def test_fill_config_dict_return_type():
    """Test that a registered function returning a dict is handled correctly."""

    @my_registry.cats.register('catsie_with_dict.v1')
    def catsie_with_dict(evil: StrictBool) -> Dict[str, bool]:
        return {'not_evil': not evil}
    config = {'test': {'@cats': 'catsie_with_dict.v1', 'evil': False}, 'foo': 10}
    result = my_registry.fill({'cfg': config}, validate=True)['cfg']['test']
    assert result['evil'] is False
    assert 'not_evil' not in result
    result = my_registry.resolve({'cfg': config}, validate=True)['cfg']['test']
    assert result['not_evil'] is True