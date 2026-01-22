import ast
import importlib
from collections import namedtuple
from typing import List, Tuple
import pytest
@pytest.mark.parametrize('module_name', ['thinc.api', 'thinc.shims', 'thinc.layers'])
def test_import_reexport_equivalency(module_name: str):
    """Tests whether a module's __all__ is equivalent to its imports. This assumes that this module is supposed to
    re-export all imported values.
    module_name (str): Module to load.
    """
    mod = importlib.import_module(module_name)
    assert set(mod.__all__) == {k for k in set((n for i in get_imports(str(mod.__file__)) for n in i.name)) if not k.startswith('_') or (module_name == 'thinc' and k == '__version__')}