import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def find_all_next(self, name=None, attrs={}, string=None, limit=None, **kwargs):
    """Find all PageElements that match the given criteria and appear
        later in the document than this PageElement.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :param limit: Stop looking after finding this many results.
        :kwargs: A dictionary of filters on attribute values.
        :return: A ResultSet containing PageElements.
        """
    _stacklevel = kwargs.pop('_stacklevel', 2)
    return self._find_all(name, attrs, string, limit, self.next_elements, _stacklevel=_stacklevel + 1, **kwargs)