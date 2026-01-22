import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def text_content(self):
    """
        Return the text content of the tag (and the text in any children).
        """
    return _collect_string_content(self)