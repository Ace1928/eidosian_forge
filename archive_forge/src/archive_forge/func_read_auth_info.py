import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
def read_auth_info(agent_file_content):
    """Loads agent authentication information from the
    specified content string, as read from an agents file.
    The returned information is suitable for passing as an argument
    to the AgentInteractor constructor.
    @param agent_file_content The agent file content (str)
    @return AuthInfo The authentication information
    @raises AgentFileFormatError when the file format is bad.
    """
    try:
        data = json.loads(agent_file_content)
        return AuthInfo(key=bakery.PrivateKey.deserialize(data['key']['private']), agents=list((Agent(url=a['url'], username=a['username']) for a in data.get('agents', []))))
    except (KeyError, ValueError, TypeError) as e:
        raise AgentFileFormatError('invalid agent file', e)