from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def create_repo_from_template(self, name: str, repo: Repository, description: Opt[str]=NotSet, include_all_branches: Opt[bool]=NotSet, private: Opt[bool]=NotSet) -> Repository:
    """
        :calls: `POST /repos/{template_owner}/{template_repo}/generate <https://docs.github.com/en/rest/reference/repos#create-a-repository-using-a-template>`_
        """
    assert isinstance(name, str), name
    assert isinstance(repo, github.Repository.Repository), repo
    assert is_optional(description, str), description
    assert is_optional(include_all_branches, bool), include_all_branches
    assert is_optional(private, bool), private
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'name': name, 'owner': self.login, 'description': description, 'include_all_branches': include_all_branches, 'private': private})
    headers, data = self._requester.requestJsonAndCheck('POST', f'/repos/{repo.owner.login}/{repo.name}/generate', input=post_parameters, headers={'Accept': 'application/vnd.github.v3+json'})
    return github.Repository.Repository(self._requester, headers, data, completed=True)