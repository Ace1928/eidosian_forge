import saml2
from saml2 import SamlBase
from saml2 import saml
def requested_attributes_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(RequestedAttributesType_, xml_string)