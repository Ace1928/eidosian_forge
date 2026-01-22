import urllib.parse
from typing import Any, Dict
from github.EnterpriseConsumedLicenses import EnterpriseConsumedLicenses
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
@property
def enterprise(self) -> str:
    return self._enterprise.value