import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
def withRequester(self, requester: Requester) -> 'NetrcAuth':
    super().withRequester(requester)
    auth = utils.get_netrc_auth(requester.base_url, raise_errors=True)
    if auth is None:
        raise RuntimeError(f'Could not get credentials from netrc for host {requester.hostname}')
    self._login, self._password = auth
    return self