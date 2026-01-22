from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
def offer_credits(self, credited: Iterable[Credit]) -> None:
    """
        Offers credit to a list of users for a vulnerability in a repository.
        Unless you are giving credit to yourself, the user having credit offered will need to explicitly accept the credit.
        :calls: `PATCH /repos/{owner}/{repo}/security-advisories/:advisory_id <https://docs.github.com/en/rest/security-advisories/repository-advisories>`
        :param credited: iterable of dict with keys "login" and "type"
        """
    assert isinstance(credited, Iterable), credited
    for credit in credited:
        AdvisoryCredit._validate_credit(credit)
    patch_parameters = {'credits': [AdvisoryCredit._to_github_dict(credit) for credit in self.credits + list(credited)]}
    headers, data = self._requester.requestJsonAndCheck('PATCH', self.url, input=patch_parameters)
    self._useAttributes(data)