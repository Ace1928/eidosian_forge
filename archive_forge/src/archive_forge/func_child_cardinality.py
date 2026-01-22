import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def child_cardinality(self, child):
    """Return the cardinality of a child element

        :param child: The name of the child element
        :return: The cardinality as a 2-tuple (min, max).
            The max value is either a number or the string "unbounded".
            The min value is always a number.
        """
    for prop, klassdef in self.c_children.values():
        if child == prop:
            if isinstance(klassdef, list):
                try:
                    _min = self.c_cardinality['min']
                except KeyError:
                    _min = 1
                try:
                    _max = self.c_cardinality['max']
                except KeyError:
                    _max = 'unbounded'
                return (_min, _max)
            else:
                return (1, 1)
    return None