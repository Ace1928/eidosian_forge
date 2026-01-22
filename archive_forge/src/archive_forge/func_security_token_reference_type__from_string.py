import saml2
from saml2 import SamlBase
def security_token_reference_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SecurityTokenReferenceType_, xml_string)