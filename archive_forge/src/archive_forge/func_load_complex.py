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
def load_complex(self, cnf):
    acs = ac_factory(cnf.get('attribute_map_dir'))
    if not acs:
        raise ConfigurationError('No attribute converters, something is wrong!!')
    self.setattr('', 'attribute_converters', acs)
    try:
        self.setattr('', 'metadata', self.load_metadata(cnf['metadata']))
    except KeyError:
        pass
    for srv, spec in cnf.get('service', {}).items():
        policy_conf = spec.get('policy')
        self.setattr(srv, 'policy', Policy(policy_conf, self.metadata))