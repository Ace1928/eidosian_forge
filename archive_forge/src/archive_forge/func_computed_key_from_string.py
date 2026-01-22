import saml2
from saml2 import SamlBase
from saml2.ws import wsaddr as wsa
from saml2.ws import wssec as wsse
from saml2.ws import wsutil as wsu
def computed_key_from_string(xml_string):
    return saml2.create_class_from_xml_string(ComputedKey, xml_string)