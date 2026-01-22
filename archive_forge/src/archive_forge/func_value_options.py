import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@property
def value_options(self):
    """
        Returns a list of all the possible values.
        """
    return [el.get('value') for el in self]