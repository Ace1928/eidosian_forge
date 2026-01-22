import saml2
from saml2 import SamlBase
def relationship_type_open_enum__from_string(xml_string):
    return saml2.create_class_from_xml_string(RelationshipTypeOpenEnum_, xml_string)