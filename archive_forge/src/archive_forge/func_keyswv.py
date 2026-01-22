import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def keyswv(self):
    """Return the keys of attributes or children that has values

        :return: list of keys
        """
    return [key for key, val in self.__dict__.items() if val]