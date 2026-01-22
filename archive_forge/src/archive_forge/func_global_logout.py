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
def global_logout(self, name_id, reason='', expire=None, sign=None, sign_alg=None, digest_alg=None):
    """More or less a layer of indirection :-/
        Bootstrapping the whole thing by finding all the IdPs that should
        be notified.

        :param name_id: The identifier of the subject that wants to be
            logged out.
        :param reason: Why the subject wants to log out
        :param expire: The latest the log out should happen.
            If this time has passed don't bother.
        :param sign: Whether the request should be signed or not.
            This also depends on what binding is used.
        :return: Depends on which binding is used:
            If the HTTP redirect binding then a HTTP redirect,
            if SOAP binding has been used the just the result of that
            conversation.
        """
    if isinstance(name_id, str):
        name_id = decode(name_id)
    logger.debug('logout request for: %s', name_id)
    entity_ids = self.users.issuers_of_info(name_id)
    return self.do_logout(name_id, entity_ids, reason, expire, sign, sign_alg=sign_alg, digest_alg=digest_alg)