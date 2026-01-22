import saml2
from saml2 import SamlBase
def subscriber_line_number_from_string(xml_string):
    return saml2.create_class_from_xml_string(SubscriberLineNumber, xml_string)