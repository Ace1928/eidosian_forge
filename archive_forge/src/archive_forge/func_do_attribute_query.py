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
def do_attribute_query(self, entityid, subject_id, attribute=None, sp_name_qualifier=None, name_qualifier=None, nameid_format=None, real_id=None, consent=None, extensions=None, sign=False, binding=BINDING_SOAP, nsprefix=None, sign_alg=None, digest_alg=None):
    """Does a attribute request to an attribute authority, this is
        by default done over SOAP.

        :param entityid: To whom the query should be sent
        :param subject_id: The identifier of the subject
        :param attribute: A dictionary of attributes and values that is
            asked for
        :param sp_name_qualifier: The unique identifier of the
            service provider or affiliation of providers for whom the
            identifier was generated.
        :param name_qualifier: The unique identifier of the identity
            provider that generated the identifier.
        :param nameid_format: The format of the name ID
        :param real_id: The identifier which is the key to this entity in the
            identity database
        :param binding: Which binding to use
        :param nsprefix: Namespace prefixes preferred before those automatically
            produced.
        :return: The attributes returned if BINDING_SOAP was used.
            HTTP args if BINDING_HTT_POST was used.
        """
    if real_id:
        response_args = {'real_id': real_id}
    else:
        response_args = {}
    if not binding:
        binding, destination = self.pick_binding('attribute_service', None, 'attribute_authority', entity_id=entityid)
    else:
        srvs = self.metadata.attribute_service(entityid, binding)
        if srvs is []:
            raise SAMLError('No attribute service support at entity')
        destination = next(locations(srvs), None)
    if binding == BINDING_SOAP:
        return self._use_soap(destination, 'attribute_query', consent=consent, extensions=extensions, sign=sign, sign_alg=sign_alg, digest_alg=digest_alg, subject_id=subject_id, attribute=attribute, sp_name_qualifier=sp_name_qualifier, name_qualifier=name_qualifier, format=nameid_format, response_args=response_args)
    elif binding == BINDING_HTTP_POST:
        mid = sid()
        query = self.create_attribute_query(destination, name_id=subject_id, attribute=attribute, message_id=mid, consent=consent, extensions=extensions, sign=sign, sign_alg=sign_alg, digest_alg=digest_alg, nsprefix=nsprefix)
        self.state[query.id] = {'entity_id': entityid, 'operation': 'AttributeQuery', 'subject_id': subject_id, 'sign': sign}
        relay_state = self._relay_state(query.id)
        return self.apply_binding(binding, str(query), destination, relay_state, sign=False, sigalg=sign_alg)
    else:
        raise SAMLError('Unsupported binding')