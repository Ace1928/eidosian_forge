import saml2
from saml2 import SamlBase
def shared_secret_dynamic_plaintext_from_string(xml_string):
    return saml2.create_class_from_xml_string(SharedSecretDynamicPlaintext, xml_string)