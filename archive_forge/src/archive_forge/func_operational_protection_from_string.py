import saml2
from saml2 import SamlBase
def operational_protection_from_string(xml_string):
    return saml2.create_class_from_xml_string(OperationalProtection, xml_string)