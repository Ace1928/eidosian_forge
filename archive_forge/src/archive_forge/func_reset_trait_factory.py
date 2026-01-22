from contextlib import contextmanager
import unittest
from traits.ctrait import CTrait
from traits.trait_converters import (
from traits.trait_factory import TraitFactory
from traits.trait_handlers import TraitCastType, TraitInstance
from traits.api import Any, Int
@contextmanager
def reset_trait_factory():
    from traits import trait_factory
    old_tfi = trait_factory._trait_factory_instances.copy()
    try:
        yield
    finally:
        trait_factory._trait_factory_instances = old_tfi