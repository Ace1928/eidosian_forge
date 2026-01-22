import saml2
from saml2 import SamlBase
def shared_secret_challenge_response_type__from_string(xml_string):
    return saml2.create_class_from_xml_string(SharedSecretChallengeResponseType_, xml_string)