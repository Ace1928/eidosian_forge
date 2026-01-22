import saml2
from saml2 import SamlBase
def attributed_ur_i__from_string(xml_string):
    return saml2.create_class_from_xml_string(AttributedURI_, xml_string)