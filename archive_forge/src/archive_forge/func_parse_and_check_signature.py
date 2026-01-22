import hashlib
from hashlib import sha1
import importlib
from itertools import chain
import json
import logging
import os
from os.path import isfile
from os.path import join
from re import compile as regex_compile
import sys
from warnings import warn as _warn
import requests
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import md
from saml2 import saml
from saml2 import samlp
from saml2 import xmldsig
from saml2 import xmlenc
from saml2.extension.algsupport import NAMESPACE as NS_ALGSUPPORT
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.extension.idpdisc import BINDING_DISCO
from saml2.extension.idpdisc import DiscoveryResponse
from saml2.extension.mdattr import NAMESPACE as NS_MDATTR
from saml2.extension.mdattr import EntityAttributes
from saml2.extension.mdrpi import NAMESPACE as NS_MDRPI
from saml2.extension.mdrpi import RegistrationInfo
from saml2.extension.mdrpi import RegistrationPolicy
from saml2.extension.mdui import NAMESPACE as NS_MDUI
from saml2.extension.mdui import Description
from saml2.extension.mdui import DisplayName
from saml2.extension.mdui import InformationURL
from saml2.extension.mdui import Logo
from saml2.extension.mdui import PrivacyStatementURL
from saml2.extension.mdui import UIInfo
from saml2.extension.shibmd import NAMESPACE as NS_SHIBMD
from saml2.extension.shibmd import Scope
from saml2.httpbase import HTTPBase
from saml2.md import NAMESPACE as NS_MD
from saml2.md import ArtifactResolutionService
from saml2.md import EntitiesDescriptor
from saml2.md import EntityDescriptor
from saml2.md import NameIDMappingService
from saml2.md import SingleSignOnService
from saml2.mdie import to_dict
from saml2.s_utils import UnknownSystemEntity
from saml2.s_utils import UnsupportedBinding
from saml2.sigver import SignatureError
from saml2.sigver import security_context
from saml2.sigver import split_len
from saml2.time_util import add_duration
from saml2.time_util import before
from saml2.time_util import instant
from saml2.time_util import str_to_time
from saml2.time_util import valid
from saml2.validate import NotValid
from saml2.validate import valid_instance
def parse_and_check_signature(self, txt):
    self.parse(txt)
    if not self.cert:
        return True
    if not self.signed():
        return True
    if self.node_name is not None:
        try:
            self.security.verify_signature(txt, node_name=self.node_name, cert_file=self.cert)
        except SignatureError as e:
            error_context = {'message': 'Failed to verify signature', 'node_name': self.node_name}
            raise SignatureError(error_context) from e
        else:
            return True

    def try_verify_signature(node_name):
        try:
            self.security.verify_signature(txt, node_name=node_name, cert_file=self.cert)
        except SignatureError:
            return False
        else:
            return True
    descriptor_names = [f'{ns}:{tag}' for ns, tag in [(EntitiesDescriptor.c_namespace, EntitiesDescriptor.c_tag), (EntityDescriptor.c_namespace, EntityDescriptor.c_tag)]]
    verified_w_descriptor_name = any((try_verify_signature(node_name) for node_name in descriptor_names))
    if not verified_w_descriptor_name:
        error_context = {'message': 'Failed to verify signature', 'descriptor_names': descriptor_names}
        raise SignatureError(error_context)
    return verified_w_descriptor_name