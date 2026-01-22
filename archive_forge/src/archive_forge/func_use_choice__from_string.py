import saml2
from saml2 import SamlBase
from saml2.schema import wsdl
def use_choice__from_string(xml_string):
    return saml2.create_class_from_xml_string(UseChoice_, xml_string)