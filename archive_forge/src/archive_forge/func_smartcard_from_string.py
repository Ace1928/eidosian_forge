import saml2
from saml2 import SamlBase
def smartcard_from_string(xml_string):
    return saml2.create_class_from_xml_string(Smartcard, xml_string)