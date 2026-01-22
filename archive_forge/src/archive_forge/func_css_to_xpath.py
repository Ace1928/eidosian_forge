import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def css_to_xpath(self, css: str, prefix: str='descendant-or-self::') -> str:
    """Translate a *group of selectors* to XPath.

        Pseudo-elements are not supported here since XPath only knows
        about "real" elements.

        :param css:
            A *group of selectors* as a string.
        :param prefix:
            This string is prepended to the XPath expression for each selector.
            The default makes selectors scoped to the context node’s subtree.
        :raises:
            :class:`~cssselect.SelectorSyntaxError` on invalid selectors,
            :class:`ExpressionError` on unknown/unsupported selectors,
            including pseudo-elements.
        :returns:
            The equivalent XPath 1.0 expression as a string.

        """
    return ' | '.join((self.selector_to_xpath(selector, prefix, translate_pseudo_elements=True) for selector in parse(css)))