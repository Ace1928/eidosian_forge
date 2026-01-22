import saml2
from saml2 import SamlBase
def expires_from_string(xml_string):
    return saml2.create_class_from_xml_string(Expires, xml_string)