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
def test_config_to_str_escapes():
    section_str = '\n        [section]\n        node1 = "^a$$"\n        node2 = "$$b$$c"\n        '
    section_dict = {'section': {'node1': '^a$', 'node2': '$b$c'}}
    cfg = Config().from_str(section_str)
    assert cfg == section_dict
    cfg = Config(section_dict)
    assert cfg == section_dict
    cfg_str = cfg.to_str()
    assert '^a$$' in cfg_str
    new_cfg = Config().from_str(cfg_str)
    assert cfg == section_dict