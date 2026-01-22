import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def create_observer(**kwargs):
    """ Convenience function for creating DictItemObserver with default values.
    """
    values = dict(notify=True, optional=False)
    values.update(kwargs)
    return DictItemObserver(**values)