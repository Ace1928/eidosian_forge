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
@pytest.mark.parametrize('cfg,is_valid', [('[a]\nb = 1\n\n[a.c]\nd = 3', True), ('[a]\nb = 1\n\n[A.c]\nd = 2', False)])
def test_cant_expand_undefined_block(cfg, is_valid):
    """Test that you can't expand a block that hasn't been created yet. This
    comes up when you typo a name, and if we allow expansion of undefined blocks,
    it's very hard to create good errors for those typos.
    """
    if is_valid:
        Config().from_str(cfg)
    else:
        with pytest.raises(ConfigValidationError):
            Config().from_str(cfg)