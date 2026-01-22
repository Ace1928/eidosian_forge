from urllib import parse
from saml2.entity import Entity
from saml2.response import VerificationError
def verify_return(self, entity_id, return_url):
    for endp in self.metadata.discovery_response(entity_id):
        if not return_url.startswith(endp['location']):
            return True
    return False