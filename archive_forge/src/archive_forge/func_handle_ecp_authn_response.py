import logging
from saml2 import BINDING_PAOS
from saml2 import BINDING_SOAP
from saml2 import element_to_extension_element
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.client_base import ACTOR
from saml2.client_base import MIME_PAOS
from saml2.ecp_client import SERVICE
from saml2.profile import ecp
from saml2.profile import paos
from saml2.response import authn_response
from saml2.schema import soapenv
from saml2.server import Server
def handle_ecp_authn_response(cls, soap_message, outstanding=None):
    rdict = soap.class_instances_from_soap_enveloped_saml_thingies(soap_message, [paos, ecp, samlp])
    _relay_state = None
    for item in rdict['header']:
        if item.c_tag == 'RelayState' and item.c_namespace == ecp.NAMESPACE:
            _relay_state = item
    response = authn_response(cls.config, cls.service_urls(), outstanding, allow_unsolicited=True)
    response.loads(f'{rdict['body']}', False, soap_message)
    response.verify()
    cls.users.add_information_about_person(response.session_info())
    return (response, _relay_state)