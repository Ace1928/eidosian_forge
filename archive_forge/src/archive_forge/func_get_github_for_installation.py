from __future__ import annotations
import urllib.parse
import warnings
from typing import Any
import deprecated
import urllib3
from urllib3 import Retry
import github
from github import Consts
from github.Auth import AppAuth
from github.GithubApp import GithubApp
from github.GithubException import GithubException
from github.Installation import Installation
from github.InstallationAuthorization import InstallationAuthorization
from github.PaginatedList import PaginatedList
from github.Requester import Requester
def get_github_for_installation(self, installation_id: int, token_permissions: dict[str, str] | None=None) -> github.Github:
    auth = self.auth.get_installation_auth(installation_id, token_permissions, self.__requester)
    return github.Github(**self.__requester.withAuth(auth).kwargs)