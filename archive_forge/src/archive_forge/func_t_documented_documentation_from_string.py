import saml2
from saml2 import SamlBase
def t_documented_documentation_from_string(xml_string):
    return saml2.create_class_from_xml_string(TDocumented_documentation, xml_string)