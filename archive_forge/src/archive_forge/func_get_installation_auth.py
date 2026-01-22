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
def get_installation_auth(self, installation_id: int, token_permissions: Optional[Dict[str, str]]=None, requester: Optional[Requester]=None) -> 'AppInstallationAuth':
    """
        Creates a github.Auth.AppInstallationAuth instance for an installation.
        :param installation_id: installation id
        :param token_permissions: optional permissions
        :param requester: optional requester with app authentication
        :return:
        """
    return AppInstallationAuth(self, installation_id, token_permissions, requester)