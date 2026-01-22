from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
def stripentities(self, keepxmlentities=False):
    """Return a copy of the text with any character or numeric entities
        replaced by the equivalent UTF-8 characters.
        
        If the `keepxmlentities` parameter is provided and evaluates to `True`,
        the core XML entities (``&amp;``, ``&apos;``, ``&gt;``, ``&lt;`` and
        ``&quot;``) are not stripped.
        
        :return: a `Markup` instance with entities removed
        :rtype: `Markup`
        :see: `genshi.util.stripentities`
        """
    return Markup(stripentities(self, keepxmlentities=keepxmlentities))