from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Branch
import github.Commit
import github.GithubObject
import github.NamedUser
import github.Tag
import github.WorkflowRun
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
def create_dispatch(self, ref: github.Branch.Branch | github.Tag.Tag | github.Commit.Commit | str, inputs: Opt[dict]=NotSet) -> bool:
    """
        :calls: `POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches <https://docs.github.com/en/rest/reference/actions#create-a-workflow-dispatch-event>`_
        """
    assert isinstance(ref, github.Branch.Branch) or isinstance(ref, github.Tag.Tag) or isinstance(ref, github.Commit.Commit) or isinstance(ref, str), ref
    assert inputs is NotSet or isinstance(inputs, dict), inputs
    if isinstance(ref, github.Branch.Branch):
        ref = ref.name
    elif isinstance(ref, github.Commit.Commit):
        ref = ref.sha
    elif isinstance(ref, github.Tag.Tag):
        ref = ref.name
    if inputs is NotSet:
        inputs = {}
    status, _, _ = self._requester.requestJson('POST', f'{self.url}/dispatches', input={'ref': ref, 'inputs': inputs})
    return status == 204