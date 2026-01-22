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
def do_assertion_id_request(self, assertion_ids, entity_id, consent=None, extensions=None, sign=False):
    srvs = self.metadata.assertion_id_request_service(entity_id, BINDING_SOAP)
    if not srvs:
        raise NoServiceDefined(f'{entity_id}: assertion_id_request_service')
    if isinstance(assertion_ids, str):
        assertion_ids = [assertion_ids]
    _id_refs = [AssertionIDRef(_id) for _id in assertion_ids]
    for destination in locations(srvs):
        res = self._use_soap(destination, 'assertion_id_request', assertion_id_refs=_id_refs, consent=consent, extensions=extensions, sign=sign)
        if res:
            return res
    return None