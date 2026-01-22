from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def edit_protection(self, strict: Opt[bool]=NotSet, contexts: Opt[list[str]]=NotSet, enforce_admins: Opt[bool]=NotSet, dismissal_users: Opt[list[str]]=NotSet, dismissal_teams: Opt[list[str]]=NotSet, dismissal_apps: Opt[list[str]]=NotSet, dismiss_stale_reviews: Opt[bool]=NotSet, require_code_owner_reviews: Opt[bool]=NotSet, required_approving_review_count: Opt[int]=NotSet, user_push_restrictions: Opt[list[str]]=NotSet, team_push_restrictions: Opt[list[str]]=NotSet, app_push_restrictions: Opt[list[str]]=NotSet, required_linear_history: Opt[bool]=NotSet, allow_force_pushes: Opt[bool]=NotSet, required_conversation_resolution: Opt[bool]=NotSet, lock_branch: Opt[bool]=NotSet, allow_fork_syncing: Opt[bool]=NotSet, users_bypass_pull_request_allowances: Opt[list[str]]=NotSet, teams_bypass_pull_request_allowances: Opt[list[str]]=NotSet, apps_bypass_pull_request_allowances: Opt[list[str]]=NotSet, block_creations: Opt[bool]=NotSet, require_last_push_approval: Opt[bool]=NotSet, allow_deletions: Opt[bool]=NotSet) -> BranchProtection:
    """
        :calls: `PUT /repos/{owner}/{repo}/branches/{branch}/protection <https://docs.github.com/en/rest/reference/repos#get-branch-protection>`_

        NOTE: The GitHub API groups strict and contexts together, both must
        be submitted. Take care to pass both as arguments even if only one is
        changing. Use edit_required_status_checks() to avoid this.
        """
    assert is_optional(strict, bool), strict
    assert is_optional_list(contexts, str), contexts
    assert is_optional(enforce_admins, bool), enforce_admins
    assert is_optional_list(dismissal_users, str), dismissal_users
    assert is_optional_list(dismissal_teams, str), dismissal_teams
    assert is_optional_list(dismissal_apps, str), dismissal_apps
    assert is_optional(dismiss_stale_reviews, bool), dismiss_stale_reviews
    assert is_optional(require_code_owner_reviews, bool), require_code_owner_reviews
    assert is_optional(required_approving_review_count, int), required_approving_review_count
    assert is_optional(required_linear_history, bool), required_linear_history
    assert is_optional(allow_force_pushes, bool), allow_force_pushes
    assert is_optional(required_conversation_resolution, bool), required_conversation_resolution
    assert is_optional(lock_branch, bool), lock_branch
    assert is_optional(allow_fork_syncing, bool), allow_fork_syncing
    assert is_optional_list(users_bypass_pull_request_allowances, str), users_bypass_pull_request_allowances
    assert is_optional_list(teams_bypass_pull_request_allowances, str), teams_bypass_pull_request_allowances
    assert is_optional_list(apps_bypass_pull_request_allowances, str), apps_bypass_pull_request_allowances
    assert is_optional(require_last_push_approval, bool), require_last_push_approval
    assert is_optional(allow_deletions, bool), allow_deletions
    post_parameters: dict[str, Any] = {}
    if is_defined(strict) or is_defined(contexts):
        if is_undefined(strict):
            strict = False
        if is_undefined(contexts):
            contexts = []
        post_parameters['required_status_checks'] = {'strict': strict, 'contexts': contexts}
    else:
        post_parameters['required_status_checks'] = None
    if is_defined(enforce_admins):
        post_parameters['enforce_admins'] = enforce_admins
    else:
        post_parameters['enforce_admins'] = None
    if is_defined(dismissal_users) or is_defined(dismissal_teams) or is_defined(dismissal_apps) or is_defined(dismiss_stale_reviews) or is_defined(require_code_owner_reviews) or is_defined(required_approving_review_count) or is_defined(users_bypass_pull_request_allowances) or is_defined(teams_bypass_pull_request_allowances) or is_defined(apps_bypass_pull_request_allowances) or is_defined(require_last_push_approval):
        post_parameters['required_pull_request_reviews'] = {}
        if is_defined(dismiss_stale_reviews):
            post_parameters['required_pull_request_reviews']['dismiss_stale_reviews'] = dismiss_stale_reviews
        if is_defined(require_code_owner_reviews):
            post_parameters['required_pull_request_reviews']['require_code_owner_reviews'] = require_code_owner_reviews
        if is_defined(required_approving_review_count):
            post_parameters['required_pull_request_reviews']['required_approving_review_count'] = required_approving_review_count
        if is_defined(require_last_push_approval):
            post_parameters['required_pull_request_reviews']['require_last_push_approval'] = require_last_push_approval
        dismissal_restrictions = {}
        if is_defined(dismissal_users):
            dismissal_restrictions['users'] = dismissal_users
        if is_defined(dismissal_teams):
            dismissal_restrictions['teams'] = dismissal_teams
        if is_defined(dismissal_apps):
            dismissal_restrictions['apps'] = dismissal_apps
        if dismissal_restrictions:
            post_parameters['required_pull_request_reviews']['dismissal_restrictions'] = dismissal_restrictions
        bypass_pull_request_allowances = {}
        if is_defined(users_bypass_pull_request_allowances):
            bypass_pull_request_allowances['users'] = users_bypass_pull_request_allowances
        if is_defined(teams_bypass_pull_request_allowances):
            bypass_pull_request_allowances['teams'] = teams_bypass_pull_request_allowances
        if is_defined(apps_bypass_pull_request_allowances):
            bypass_pull_request_allowances['apps'] = apps_bypass_pull_request_allowances
        if bypass_pull_request_allowances:
            post_parameters['required_pull_request_reviews']['bypass_pull_request_allowances'] = bypass_pull_request_allowances
    else:
        post_parameters['required_pull_request_reviews'] = None
    if is_defined(user_push_restrictions) or is_defined(team_push_restrictions) or is_defined(app_push_restrictions):
        if is_undefined(user_push_restrictions):
            user_push_restrictions = []
        if is_undefined(team_push_restrictions):
            team_push_restrictions = []
        if is_undefined(app_push_restrictions):
            app_push_restrictions = []
        post_parameters['restrictions'] = {'users': user_push_restrictions, 'teams': team_push_restrictions, 'apps': app_push_restrictions}
    else:
        post_parameters['restrictions'] = None
    if is_defined(required_linear_history):
        post_parameters['required_linear_history'] = required_linear_history
    else:
        post_parameters['required_linear_history'] = None
    if is_defined(allow_force_pushes):
        post_parameters['allow_force_pushes'] = allow_force_pushes
    else:
        post_parameters['allow_force_pushes'] = None
    if is_defined(required_conversation_resolution):
        post_parameters['required_conversation_resolution'] = required_conversation_resolution
    else:
        post_parameters['required_conversation_resolution'] = None
    if is_defined(lock_branch):
        post_parameters['lock_branch'] = lock_branch
    else:
        post_parameters['lock_branch'] = None
    if is_defined(allow_fork_syncing):
        post_parameters['allow_fork_syncing'] = allow_fork_syncing
    else:
        post_parameters['allow_fork_syncing'] = None
    if is_defined(block_creations):
        post_parameters['block_creations'] = block_creations
    else:
        post_parameters['block_creations'] = None
    if is_defined(allow_deletions):
        post_parameters['allow_deletions'] = allow_deletions
    else:
        post_parameters['allow_deletions'] = None
    headers, data = self._requester.requestJsonAndCheck('PUT', self.protection_url, headers={'Accept': Consts.mediaTypeRequireMultipleApprovingReviews}, input=post_parameters)
    return github.BranchProtection.BranchProtection(self._requester, headers, data, completed=True)