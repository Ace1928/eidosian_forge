import copy
import importlib
import logging
from logging.config import dictConfig as configure_logging_by_dict
import logging.handlers
import os
import re
import sys
from warnings import warn as _warn
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2.assertion import Policy
from saml2.attribute_converter import ac_factory
from saml2.mdstore import MetadataStore
from saml2.saml import NAME_FORMAT_URI
from saml2.virtual_org import VirtualOrg
def do_extensions(self, extensions):
    for key, val in extensions.items():
        self.extensions[key] = val