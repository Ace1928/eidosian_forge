from urllib import parse
from saml2.entity import Entity
from saml2.response import VerificationError
@staticmethod
def create_discovery_service_response(return_url=None, returnIDParam='entityID', entity_id=None, **kwargs):
    if return_url is None:
        return_url = kwargs['return']
    if entity_id:
        qp = parse.urlencode({returnIDParam: entity_id})
        part = parse.urlparse(return_url)
        if part.query:
            return_url = f'{return_url}&{qp}'
        else:
            return_url = f'{return_url}?{qp}'
    return return_url