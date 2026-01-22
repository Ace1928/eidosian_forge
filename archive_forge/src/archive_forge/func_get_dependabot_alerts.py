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
def get_dependabot_alerts(self, state: Opt[str]=NotSet, severity: Opt[str]=NotSet, ecosystem: Opt[str]=NotSet, package: Opt[str]=NotSet, scope: Opt[str]=NotSet, sort: Opt[str]=NotSet, direction: Opt[str]=NotSet) -> PaginatedList[OrganizationDependabotAlert]:
    """
        :calls: `GET /orgs/{org}/dependabot/alerts <https://docs.github.com/en/rest/dependabot/alerts#list-dependabot-alerts-for-an-organization>`_
        :param state: Optional string
        :param severity: Optional string
        :param ecosystem: Optional string
        :param package: Optional string
        :param scope: Optional string
        :param sort: Optional string
        :param direction: Optional string
        :rtype: :class:`PaginatedList` of :class:`github.DependabotAlert.DependabotAlert`
        """
    allowed_states = ['auto_dismissed', 'dismissed', 'fixed', 'open']
    allowed_severities = ['low', 'medium', 'high', 'critical']
    allowed_ecosystems = ['composer', 'go', 'maven', 'npm', 'nuget', 'pip', 'pub', 'rubygems', 'rust']
    allowed_scopes = ['development', 'runtime']
    allowed_sorts = ['created', 'updated']
    allowed_directions = ['asc', 'desc']
    assert state in allowed_states + [NotSet], f'State can be one of {', '.join(allowed_states)}'
    assert severity in allowed_severities + [NotSet], f'Severity can be one of {', '.join(allowed_severities)}'
    assert ecosystem in allowed_ecosystems + [NotSet], f'Ecosystem can be one of {', '.join(allowed_ecosystems)}'
    assert scope in allowed_scopes + [NotSet], f'Scope can be one of {', '.join(allowed_scopes)}'
    assert sort in allowed_sorts + [NotSet], f'Sort can be one of {', '.join(allowed_sorts)}'
    assert direction in allowed_directions + [NotSet], f'Direction can be one of {', '.join(allowed_directions)}'
    url_parameters = NotSet.remove_unset_items({'state': state, 'severity': severity, 'ecosystem': ecosystem, 'package': package, 'scope': scope, 'sort': sort, 'direction': direction})
    return PaginatedList(github.OrganizationDependabotAlert.OrganizationDependabotAlert, self._requester, f'{self.url}/dependabot/alerts', url_parameters)