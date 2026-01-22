import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
def extensions_from_string(xml_string):
    return saml2.create_class_from_xml_string(Extensions, xml_string)