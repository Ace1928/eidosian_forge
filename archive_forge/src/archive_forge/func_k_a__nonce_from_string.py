import saml2
from saml2 import SamlBase
from saml2 import xmldsig as ds
def k_a__nonce_from_string(xml_string):
    return saml2.create_class_from_xml_string(KA_Nonce, xml_string)