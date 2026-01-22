import saml2
from saml2 import SamlBase
def timestamp_from_string(xml_string):
    return saml2.create_class_from_xml_string(Timestamp, xml_string)