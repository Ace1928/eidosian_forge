import json
from six.moves import urllib
from google_reauth import errors
def get_challenges(http_request, supported_challenge_types, access_token, requested_scopes=None):
    """Does initial request to reauth API to get the challenges.

    Args:
        http_request (Callable): callable to run http requests. Accepts uri,
            method, body and headers. Returns a tuple: (response, content)
        supported_challenge_types (Sequence[str]): list of challenge names
            supported by the manager.
        access_token (str): Access token with reauth scopes.
        requested_scopes (list[str]): Authorized scopes for the credentials.

    Returns:
        dict: The response from the reauth API.
    """
    body = {'supportedChallengeTypes': supported_challenge_types}
    if requested_scopes:
        body['oauthScopesForDomainPolicyLookup'] = requested_scopes
    return _endpoint_request(http_request, ':start', body, access_token)