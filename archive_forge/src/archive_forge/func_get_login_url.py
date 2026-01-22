from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
import github.AccessToken
import github.Auth
from github.GithubException import BadCredentialsException, GithubException
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
def get_login_url(self, redirect_uri: str | None=None, state: str | None=None, login: str | None=None) -> str:
    """Return the URL you need to redirect a user to in order to authorize your App."""
    parameters = {'client_id': self.client_id}
    if redirect_uri is not None:
        assert isinstance(redirect_uri, str), redirect_uri
        parameters['redirect_uri'] = redirect_uri
    if state is not None:
        assert isinstance(state, str), state
        parameters['state'] = state
    if login is not None:
        assert isinstance(login, str), login
        parameters['login'] = login
    query = urllib.parse.urlencode(parameters)
    base_url = 'https://github.com/login/oauth/authorize'
    return f'{base_url}?{query}'