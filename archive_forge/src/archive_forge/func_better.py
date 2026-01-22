from saml2 import extension_elements_to_elements
from saml2.authn_context import ippword
from saml2.authn_context import mobiletwofactor
from saml2.authn_context import ppt
from saml2.authn_context import pword
from saml2.authn_context import sslcert
from saml2.saml import AuthnContext
from saml2.saml import AuthnContextClassRef
from saml2.samlp import RequestedAuthnContext
@staticmethod
def better(a, b):
    return b > a