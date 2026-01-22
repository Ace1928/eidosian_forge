from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.EnvironmentProtectionRuleReviewer
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def wait_timer(self) -> int:
    return self._wait_timer.value