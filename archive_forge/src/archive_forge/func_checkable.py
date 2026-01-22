import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@property
def checkable(self):
    """
        Boolean: can this element be checked?
        """
    return self.type in ('checkbox', 'radio')