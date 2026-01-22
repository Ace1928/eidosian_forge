from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import NotRequired, TypedDict
import github.Commit
import github.File
import github.IssueComment
import github.IssueEvent
import github.Label
import github.Milestone
import github.NamedUser
import github.PaginatedList
import github.PullRequestComment
import github.PullRequestMergeStatus
import github.PullRequestPart
import github.PullRequestReview
import github.Team
from github import Consts
from github.GithubObject import (
from github.Issue import Issue
from github.PaginatedList import PaginatedList
def enable_automerge(self, merge_method: Opt[str]='MERGE', author_email: Opt[str]=NotSet, client_mutation_id: Opt[str]=NotSet, commit_body: Opt[str]=NotSet, commit_headline: Opt[str]=NotSet, expected_head_oid: Opt[str]=NotSet) -> dict[str, Any]:
    """
        :calls: `POST /graphql <https://docs.github.com/en/graphql>`_ with a mutation to enable pull request auto merge
        <https://docs.github.com/en/graphql/reference/mutations#enablepullrequestautomerge>
        """
    assert is_optional(author_email, str), author_email
    assert is_optional(client_mutation_id, str), client_mutation_id
    assert is_optional(commit_body, str), commit_body
    assert is_optional(commit_headline, str), commit_headline
    assert is_optional(expected_head_oid, str), expected_head_oid
    assert isinstance(merge_method, str) and merge_method in ['MERGE', 'REBASE', 'SQUASH'], merge_method
    variables = {'pullRequestId': self.node_id, 'authorEmail': author_email, 'clientMutationId': client_mutation_id, 'commitBody': commit_body, 'commitHeadline': commit_headline, 'expectedHeadOid': expected_head_oid, 'mergeMethod': merge_method}
    _, data = self._requester.graphql_named_mutation(mutation_name='enable_pull_request_auto_merge', variables=NotSet.remove_unset_items(variables), output='actor { avatarUrl login resourcePath url } clientMutationId')
    return data