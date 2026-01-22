from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def filled_seats(self) -> int:
    return self._filled_seats.value