import saml2
from saml2 import SamlBase
def fault_faultcode_from_string(xml_string):
    return saml2.create_class_from_xml_string(Fault_faultcode, xml_string)