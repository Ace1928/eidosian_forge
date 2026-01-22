from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.NotificationSubject
import github.PullRequest
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def get_pull_request(self) -> github.PullRequest.PullRequest:
    headers, data = self._requester.requestJsonAndCheck('GET', self.subject.url)
    return github.PullRequest.PullRequest(self._requester, headers, data, completed=True)