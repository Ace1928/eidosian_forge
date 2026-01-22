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
def create_git_commit(self, message: str, tree: GitTree, parents: list[GitCommit], author: Opt[InputGitAuthor]=NotSet, committer: Opt[InputGitAuthor]=NotSet) -> GitCommit:
    """
        :calls: `POST /repos/{owner}/{repo}/git/commits <https://docs.github.com/en/rest/reference/git#commits>`_
        :param message: string
        :param tree: :class:`github.GitTree.GitTree`
        :param parents: list of :class:`github.GitCommit.GitCommit`
        :param author: :class:`github.InputGitAuthor.InputGitAuthor`
        :param committer: :class:`github.InputGitAuthor.InputGitAuthor`
        :rtype: :class:`github.GitCommit.GitCommit`
        """
    assert isinstance(message, str), message
    assert isinstance(tree, github.GitTree.GitTree), tree
    assert all((isinstance(element, github.GitCommit.GitCommit) for element in parents)), parents
    assert is_optional(author, github.InputGitAuthor), author
    assert is_optional(committer, github.InputGitAuthor), committer
    post_parameters: dict[str, Any] = {'message': message, 'tree': tree._identity, 'parents': [element._identity for element in parents]}
    if is_defined(author):
        post_parameters['author'] = author._identity
    if is_defined(committer):
        post_parameters['committer'] = committer._identity
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/git/commits', input=post_parameters)
    return github.GitCommit.GitCommit(self._requester, headers, data, completed=True)