import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
def header_from_string(xml_string):
    return saml2.create_class_from_xml_string(Header, xml_string)