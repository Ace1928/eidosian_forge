from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def visual_studio_subscription_email(self) -> str:
    self._completeIfNotSet(self._visual_studio_subscription_email)
    return self._visual_studio_subscription_email.value