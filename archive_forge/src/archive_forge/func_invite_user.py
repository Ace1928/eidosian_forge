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
def invite_user(self, user: Opt[NamedUser]=NotSet, email: Opt[str]=NotSet, role: Opt[str]=NotSet, teams: Opt[list[Team]]=NotSet) -> None:
    """
        :calls: `POST /orgs/{org}/invitations <https://docs.github.com/en/rest/reference/orgs#members>`_
        :param user: :class:`github.NamedUser.NamedUser`
        :param email: string
        :param role: string
        :param teams: array of :class:`github.Team.Team`
        :rtype: None
        """
    assert is_optional(user, github.NamedUser.NamedUser), user
    assert is_optional(email, str), email
    assert is_defined(email) != is_defined(user), 'specify only one of email or user'
    assert is_undefined(role) or role in ['admin', 'direct_member', 'billing_manager'], role
    assert is_optional_list(teams, github.Team.Team), teams
    parameters: dict[str, Any] = NotSet.remove_unset_items({'email': email, 'role': role})
    if is_defined(user):
        parameters['invitee_id'] = user.id
    if is_defined(teams):
        parameters['team_ids'] = [t.id for t in teams]
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/invitations', headers={'Accept': Consts.mediaTypeOrganizationInvitationPreview}, input=parameters)