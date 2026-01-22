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
def get_team_membership(self, member: str | NamedUser) -> Membership:
    """
        :calls: `GET /orgs/{org}/memberships/team/{team_id}/{username} <https://docs.github.com/en/rest/reference/teams#get-team-membership-for-a-user>`_
        """
    assert isinstance(member, str) or isinstance(member, github.NamedUser.NamedUser), member
    if isinstance(member, github.NamedUser.NamedUser):
        member = member._identity
    else:
        member = urllib.parse.quote(member)
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/memberships/{member}')
    return github.Membership.Membership(self._requester, headers, data, completed=True)