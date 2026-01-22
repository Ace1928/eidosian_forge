from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.GithubObject
import github.HookDelivery
import github.NamedUser
import github.OrganizationDependabotAlert
import github.OrganizationSecret
import github.OrganizationVariable
import github.Plan
import github.Project
import github.Repository
import github.Team
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_secret(self, secret_name: str, secret_type: str='actions') -> OrganizationSecret:
    """
        :calls: 'GET /orgs/{org}/{secret_type}/secrets/{secret_name} <https://docs.github.com/en/rest/actions/secrets#get-an-organization-secret>`_
        :param secret_name: string
        :param secret_type: string options actions or dependabot
        :rtype: github.OrganizationSecret.OrganizationSecret
        """
    assert isinstance(secret_name, str), secret_name
    return github.OrganizationSecret.OrganizationSecret(requester=self._requester, headers={}, attributes={'url': f'{self.url}/{secret_type}/secrets/{urllib.parse.quote(secret_name)}'}, completed=False)