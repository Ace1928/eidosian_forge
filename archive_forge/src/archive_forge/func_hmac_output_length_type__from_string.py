import saml2
from saml2 import SamlBase
def hmac_output_length_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(HMACOutputLengthType_, xml_string)