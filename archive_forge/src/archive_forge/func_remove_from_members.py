from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.NamedUser
import github.Organization
import github.PaginatedList
import github.Repository
import github.TeamDiscussion
from github import Consts
from github.GithubException import UnknownObjectException
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
def remove_from_members(self, member: NamedUser) -> None:
    """
        This API call is deprecated. Use `remove_membership` instead:
        https://docs.github.com/en/rest/reference/teams#add-or-update-team-membership-for-a-user-legacy

        :calls: `DELETE /teams/{id}/members/{user} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/members/{member._identity}')