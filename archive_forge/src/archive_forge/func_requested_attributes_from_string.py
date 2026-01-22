import saml2
from saml2 import SamlBase
from saml2 import saml
def requested_attributes_from_string(xml_string):
    return saml2.create_class_from_xml_string(RequestedAttributes, xml_string)