import saml2
from saml2 import SamlBase
def binary_security_token_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(BinarySecurityTokenType_, xml_string)