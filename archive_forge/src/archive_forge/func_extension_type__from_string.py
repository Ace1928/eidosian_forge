import saml2
from saml2 import SamlBase
def extension_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(ExtensionType_, xml_string)