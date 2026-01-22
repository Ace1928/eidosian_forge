import saml2
from saml2 import SamlBase
def signature_properties_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SignaturePropertiesType_, xml_string)