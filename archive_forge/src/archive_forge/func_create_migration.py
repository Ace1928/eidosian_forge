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
def create_migration(self, repos: list[Repository] | tuple[Repository], lock_repositories: Opt[bool]=NotSet, exclude_attachments: Opt[bool]=NotSet) -> Migration:
    """
        :calls: `POST /user/migrations <https://docs.github.com/en/rest/reference/migrations>`_
        """
    assert isinstance(repos, (list, tuple)), repos
    assert all((isinstance(repo, str) for repo in repos)), repos
    assert is_optional(lock_repositories, bool), lock_repositories
    assert is_optional(exclude_attachments, bool), exclude_attachments
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'repositories': repos, 'lock_repositories': lock_repositories, 'exclude_attachments': exclude_attachments})
    headers, data = self._requester.requestJsonAndCheck('POST', '/user/migrations', input=post_parameters, headers={'Accept': Consts.mediaTypeMigrationPreview})
    return github.Migration.Migration(self._requester, headers, data, completed=True)