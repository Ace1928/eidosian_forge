import saml2
from saml2 import SamlBase
def t_param__from_string(xml_string):
    return saml2.create_class_from_xml_string(TParam_, xml_string)