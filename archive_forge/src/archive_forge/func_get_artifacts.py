from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
import github.GitCommit
import github.PullRequest
import github.WorkflowJob
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
def get_artifacts(self) -> PaginatedList[Artifact]:
    return PaginatedList(github.Artifact.Artifact, self._requester, self._artifacts_url.value, None, list_item='artifacts')