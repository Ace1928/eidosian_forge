import saml2
from saml2 import SamlBase
def t_definitions__from_string(xml_string):
    return saml2.create_class_from_xml_string(TDefinitions_, xml_string)