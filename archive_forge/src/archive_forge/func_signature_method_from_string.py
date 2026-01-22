import saml2
from saml2 import SamlBase
def signature_method_from_string(xml_string):
    return saml2.create_class_from_xml_string(SignatureMethod, xml_string)