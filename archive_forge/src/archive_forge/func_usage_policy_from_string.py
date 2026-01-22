import saml2
from saml2 import SamlBase
from saml2 import md
def usage_policy_from_string(xml_string):
    return saml2.create_class_from_xml_string(UsagePolicy, xml_string)