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
def remove_membership(self, member: NamedUser) -> None:
    """
        :calls: `DELETE /teams/{team_id}/memberships/{username} <https://docs.github.com/en/rest/reference/teams#remove-team-membership-for-a-user>`_
        """
    assert isinstance(member, github.NamedUser.NamedUser), member
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/memberships/{member._identity}')