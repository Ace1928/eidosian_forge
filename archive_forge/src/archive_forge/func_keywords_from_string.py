import saml2
from saml2 import SamlBase
from saml2 import md
def keywords_from_string(xml_string):
    return saml2.create_class_from_xml_string(Keywords, xml_string)