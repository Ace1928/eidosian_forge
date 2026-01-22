import saml2
from saml2 import SamlBase
def x509_issuer_name_from_string(xml_string):
    return saml2.create_class_from_xml_string(X509IssuerName, xml_string)