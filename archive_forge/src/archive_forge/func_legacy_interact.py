import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
def legacy_interact(self, client, location, visit_url):
    """Implement LegacyInteractor.legacy_interact by obtaining
        the discharge macaroon using the client's private key
        """
    agent = self._find_agent(location)
    client = copy.copy(client)
    client.key = self._auth_info.key
    resp = client.request(method='POST', url=visit_url, json={'username': agent.username, 'public_key': str(self._auth_info.key.public_key)})
    if resp.status_code != 200:
        raise httpbakery.InteractionError('cannot acquire agent macaroon from {}: {} (response body: {!r})'.format(visit_url, resp.status_code, resp.text))
    if not resp.json().get('agent_login', False):
        raise httpbakery.InteractionError('agent login failed')