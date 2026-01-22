from __future__ import annotations
from datetime import datetime
from typing import Any
from github.CVSS import CVSS
from github.CWE import CWE
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def identifiers(self) -> list[dict]:
    return self._identifiers.value