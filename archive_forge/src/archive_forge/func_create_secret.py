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
def create_secret(self, secret_name: str, unencrypted_value: str, visibility: str='all', selected_repositories: Opt[list[github.Repository.Repository]]=NotSet) -> github.OrganizationSecret.OrganizationSecret:
    """
        :calls: `PUT /orgs/{org}/actions/secrets/{secret_name} <https://docs.github.com/en/rest/actions/secrets#create-or-update-an-organization-secret>`_
        """
    assert isinstance(secret_name, str), secret_name
    assert isinstance(unencrypted_value, str), unencrypted_value
    assert isinstance(visibility, str), visibility
    if visibility == 'selected':
        assert isinstance(selected_repositories, list) and all((isinstance(element, github.Repository.Repository) for element in selected_repositories)), selected_repositories
    else:
        assert selected_repositories is NotSet
    public_key = self.get_public_key()
    payload = public_key.encrypt(unencrypted_value)
    put_parameters: dict[str, Any] = {'key_id': public_key.key_id, 'encrypted_value': payload, 'visibility': visibility}
    if is_defined(selected_repositories):
        put_parameters['selected_repository_ids'] = [element.id for element in selected_repositories]
    self._requester.requestJsonAndCheck('PUT', f'{self.url}/actions/secrets/{urllib.parse.quote(secret_name)}', input=put_parameters)
    return github.OrganizationSecret.OrganizationSecret(requester=self._requester, headers={}, attributes={'name': secret_name, 'visibility': visibility, 'selected_repositories_url': f'{self.url}/actions/secrets/{urllib.parse.quote(secret_name)}/repositories', 'url': f'{self.url}/actions/secrets/{urllib.parse.quote(secret_name)}'}, completed=False)