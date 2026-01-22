from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def edit_required_pull_request_reviews(self, dismissal_users: Opt[list[str]]=NotSet, dismissal_teams: Opt[list[str]]=NotSet, dismissal_apps: Opt[list[str]]=NotSet, dismiss_stale_reviews: Opt[bool]=NotSet, require_code_owner_reviews: Opt[bool]=NotSet, required_approving_review_count: Opt[int]=NotSet, require_last_push_approval: Opt[bool]=NotSet) -> RequiredStatusChecks:
    """
        :calls: `PATCH /repos/{owner}/{repo}/branches/{branch}/protection/required_pull_request_reviews <https://docs.github.com/en/rest/reference/repos#branches>`_
        """
    assert is_optional_list(dismissal_users, str), dismissal_users
    assert is_optional_list(dismissal_teams, str), dismissal_teams
    assert is_optional(dismiss_stale_reviews, bool), dismiss_stale_reviews
    assert is_optional(require_code_owner_reviews, bool), require_code_owner_reviews
    assert is_optional(required_approving_review_count, int), required_approving_review_count
    assert is_optional(require_last_push_approval, bool), require_last_push_approval
    post_parameters: dict[str, Any] = NotSet.remove_unset_items({'dismiss_stale_reviews': dismiss_stale_reviews, 'require_code_owner_reviews': require_code_owner_reviews, 'required_approving_review_count': required_approving_review_count, 'require_last_push_approval': require_last_push_approval})
    dismissal_restrictions: dict[str, Any] = NotSet.remove_unset_items({'users': dismissal_users, 'teams': dismissal_teams, 'apps': dismissal_apps})
    if dismissal_restrictions:
        post_parameters['dismissal_restrictions'] = dismissal_restrictions
    headers, data = self._requester.requestJsonAndCheck('PATCH', f'{self.protection_url}/required_pull_request_reviews', headers={'Accept': Consts.mediaTypeRequireMultipleApprovingReviews}, input=post_parameters)
    return github.RequiredStatusChecks.RequiredStatusChecks(self._requester, headers, data, completed=True)