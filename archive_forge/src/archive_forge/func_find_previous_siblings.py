import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def find_previous_siblings(self, name=None, attrs={}, string=None, limit=None, **kwargs):
    """Returns all siblings to this PageElement that match the
        given criteria and appear earlier in the document.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: A dictionary of filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :param limit: Stop looking after finding this many results.
        :kwargs: A dictionary of filters on attribute values.
        :return: A ResultSet of PageElements.
        :rtype: bs4.element.ResultSet
        """
    _stacklevel = kwargs.pop('_stacklevel', 2)
    return self._find_all(name, attrs, string, limit, self.previous_siblings, _stacklevel=_stacklevel + 1, **kwargs)