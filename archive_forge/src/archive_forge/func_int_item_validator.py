import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def int_item_validator(item):
    """
    An item_validator for TraitList that checks that the item is an int
    or integer-like object (e.g., any object whose type provides __index__).

    Parameters
    ----------
    item : object
        Proposed item to add to the list.

    Returns
    -------
    validated_item : object
        Actual item to add to the list.

    Raises
    ------
    TraitError
        If the item is not valid.
    """
    try:
        return int(operator.index(item))
    except TypeError:
        raise TraitError('Value {} is not a valid integer'.format(item))