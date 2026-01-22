import saml2
from saml2 import SamlBase
def restricted_password_type__length_from_string(xml_string):
    return saml2.create_class_from_xml_string(RestrictedPasswordType_Length, xml_string)