import saml2
from saml2 import SamlBase
def t_port_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(TPortType_, xml_string)