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
def test_fill_config_overrides():
    config = {'cfg': {'one': 1, 'two': {'three': {'@cats': 'catsie.v1', 'evil': True, 'cute': False}}}}
    overrides = {'cfg.two.three.evil': False}
    result = my_registry.fill(config, overrides=overrides, validate=True)
    assert result['cfg']['two']['three']['evil'] is False
    overrides = {'cfg.two.three': 3}
    result = my_registry.fill(config, overrides=overrides, validate=True)
    assert result['cfg']['two']['three'] == 3
    overrides = {'cfg': {'one': {'@cats': 'catsie.v1', 'evil': False}, 'two': None}}
    result = my_registry.fill(config, overrides=overrides)
    assert result['cfg']['two'] is None
    assert result['cfg']['one']['@cats'] == 'catsie.v1'
    assert result['cfg']['one']['evil'] is False
    assert result['cfg']['one']['cute'] is True
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.two.three.evil': 20}
        my_registry.fill(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg': {'one': {'@cats': 'catsie.v1'}, 'two': None}}
        my_registry.fill(config, overrides=overrides)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.two.three.evil': False, 'two.four': True}
        my_registry.fill(config, overrides=overrides, validate=True)
    with pytest.raises(ConfigValidationError):
        overrides = {'cfg.five': False}
        my_registry.fill(config, overrides=overrides, validate=True)