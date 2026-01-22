import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def make_instance(klass, spec, base64encode=False):
    """
    Constructs a class instance containing the specified information

    :param klass: The class
    :param spec: Information to be placed in the instance (a dictionary)
    :return: The instance
    """
    return klass().loadd(spec, base64encode)