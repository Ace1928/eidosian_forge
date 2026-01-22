from __future__ import annotations
import collections
import urllib.parse
from base64 import b64encode
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.AdvisoryCredit
import github.AdvisoryVulnerability
import github.Artifact
import github.AuthenticatedUser
import github.Autolink
import github.Branch
import github.CheckRun
import github.CheckSuite
import github.Clones
import github.CodeScanAlert
import github.Commit
import github.CommitComment
import github.Comparison
import github.ContentFile
import github.DependabotAlert
import github.Deployment
import github.Download
import github.Environment
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
import github.EnvironmentProtectionRuleReviewer
import github.Event
import github.GitBlob
import github.GitCommit
import github.GithubObject
import github.GitRef
import github.GitRelease
import github.GitReleaseAsset
import github.GitTag
import github.GitTree
import github.Hook
import github.HookDelivery
import github.Invitation
import github.Issue
import github.IssueComment
import github.IssueEvent
import github.Label
import github.License
import github.Milestone
import github.NamedUser
import github.Notification
import github.Organization
import github.PaginatedList
import github.Path
import github.Permissions
import github.Project
import github.PublicKey
import github.PullRequest
import github.PullRequestComment
import github.Referrer
import github.RepositoryAdvisory
import github.RepositoryKey
import github.RepositoryPreferences
import github.Secret
import github.SelfHostedActionsRunner
import github.SourceImport
import github.Stargazer
import github.StatsCodeFrequency
import github.StatsCommitActivity
import github.StatsContributor
import github.StatsParticipation
import github.StatsPunchCard
import github.Tag
import github.Team
import github.Variable
import github.View
import github.Workflow
import github.WorkflowRun
from github import Consts
from github.Environment import Environment
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def create_environment(self, environment_name: str, wait_timer: int=0, reviewers: list[ReviewerParams]=[], deployment_branch_policy: EnvironmentDeploymentBranchPolicyParams | None=None) -> Environment:
    """
        :calls: `PUT /repositories/{self._repository.id}/environments/{self.environment_name}/environments/{environment_name} <https://docs.github.com/en/rest/reference/deployments#create-or-update-an-environment>`_
        :param environment_name: string
        :param wait_timer: int
        :param reviews: List[:class:github.EnvironmentDeploymentBranchPolicy.EnvironmentDeploymentBranchPolicyParams]
        :param deployment_branch_policy: Optional[:class:github.EnvironmentDeploymentBranchPolicy.EnvironmentDeploymentBranchPolicyParams`]
        :rtype: :class:`github.Environment.Environment`
        """
    assert isinstance(environment_name, str), environment_name
    assert isinstance(wait_timer, int)
    assert isinstance(reviewers, list)
    assert all([isinstance(reviewer, github.EnvironmentProtectionRuleReviewer.ReviewerParams) for reviewer in reviewers])
    assert isinstance(deployment_branch_policy, github.EnvironmentDeploymentBranchPolicy.EnvironmentDeploymentBranchPolicyParams) or deployment_branch_policy is None
    environment_name = urllib.parse.quote(environment_name)
    put_parameters = {'wait_timer': wait_timer, 'reviewers': [reviewer._asdict() for reviewer in reviewers], 'deployment_branch_policy': deployment_branch_policy._asdict() if deployment_branch_policy else None}
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/environments/{environment_name}', input=put_parameters)
    data['environments_url'] = f'/repositories/{self.id}/environments'
    return Environment(self._requester, headers, data, completed=True)