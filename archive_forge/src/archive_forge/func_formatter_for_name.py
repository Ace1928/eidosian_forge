import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def formatter_for_name(self, formatter):
    """Look up or create a Formatter for the given identifier,
        if necessary.

        :param formatter: Can be a Formatter object (used as-is), a
            function (used as the entity substitution hook for an
            XMLFormatter or HTMLFormatter), or a string (used to look
            up an XMLFormatter or HTMLFormatter in the appropriate
            registry.
        """
    if isinstance(formatter, Formatter):
        return formatter
    if self._is_xml:
        c = XMLFormatter
    else:
        c = HTMLFormatter
    if isinstance(formatter, Callable):
        return c(entity_substitution=formatter)
    return c.REGISTRY[formatter]