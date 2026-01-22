import saml2
from saml2 import SamlBase
def t_message_part_from_string(xml_string):
    return saml2.create_class_from_xml_string(TMessage_part, xml_string)