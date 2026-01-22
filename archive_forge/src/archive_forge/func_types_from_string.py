import saml2
from saml2 import SamlBase
def types_from_string(xml_string):
    return saml2.create_class_from_xml_string(Types, xml_string)