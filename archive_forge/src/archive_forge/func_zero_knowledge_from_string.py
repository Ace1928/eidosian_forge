import saml2
from saml2 import SamlBase
def zero_knowledge_from_string(xml_string):
    return saml2.create_class_from_xml_string(ZeroKnowledge, xml_string)