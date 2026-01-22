import saml2
from saml2 import SamlBase
def t_message__from_string(xml_string):
    return saml2.create_class_from_xml_string(TMessage_, xml_string)