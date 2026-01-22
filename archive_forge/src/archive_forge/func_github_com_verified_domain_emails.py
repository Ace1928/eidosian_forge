from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_verified_domain_emails(self) -> list:
    self._completeIfNotSet(self._github_com_verified_domain_emails)
    return self._github_com_verified_domain_emails.value