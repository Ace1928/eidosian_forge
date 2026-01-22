import saml2
from saml2 import SamlBase
def x509_ski_from_string(xml_string):
    return saml2.create_class_from_xml_string(X509SKI, xml_string)