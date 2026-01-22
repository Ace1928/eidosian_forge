from saml2 import extension_elements_to_elements
from saml2.authn_context import ippword
from saml2.authn_context import mobiletwofactor
from saml2.authn_context import ppt
from saml2.authn_context import pword
from saml2.authn_context import sslcert
from saml2.saml import AuthnContext
from saml2.saml import AuthnContextClassRef
from saml2.samlp import RequestedAuthnContext
def requested_authn_context(class_ref, comparison='minimum'):
    if not isinstance(class_ref, list):
        class_ref = [class_ref]
    return RequestedAuthnContext(authn_context_class_ref=[AuthnContextClassRef(text=i) for i in class_ref], comparison=comparison)