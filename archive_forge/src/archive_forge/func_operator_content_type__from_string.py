import saml2
from saml2 import SamlBase
def operator_content_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(OperatorContentType_, xml_string)