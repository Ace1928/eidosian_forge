import logging
import saml2
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2.client_base import Base
from saml2.client_base import LogoutError
from saml2.client_base import NoServiceDefined
from saml2.client_base import SignOnError
from saml2.httpbase import HTTPError
from saml2.ident import code
from saml2.ident import decode
from saml2.mdstore import locations
from saml2.s_utils import sid
from saml2.s_utils import status_message_factory
from saml2.s_utils import success_status_factory
from saml2.saml import AssertionIDRef
from saml2.samlp import STATUS_REQUEST_DENIED
from saml2.samlp import STATUS_UNKNOWN_PRINCIPAL
from saml2.time_util import not_on_or_after
def do_authz_decision_query(self, entity_id, action, subject_id, nameid_format, evidence=None, resource=None, sp_name_qualifier=None, name_qualifier=None, consent=None, extensions=None, sign=False):
    subject = saml.Subject(name_id=saml.NameID(text=subject_id, format=nameid_format, sp_name_qualifier=sp_name_qualifier, name_qualifier=name_qualifier))
    srvs = self.metadata.authz_service(entity_id, BINDING_SOAP)
    for dest in locations(srvs):
        resp = self._use_soap(dest, 'authz_decision_query', action=action, evidence=evidence, resource=resource, subject=subject)
        if resp:
            return resp
    return None