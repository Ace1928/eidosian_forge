from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt
def request_cve(self) -> None:
    """
        Requests a CVE for the advisory.
        :calls: `POST /repos/{owner}/{repo}/security-advisories/{ghsa_id}/cve <https://docs.github.com/en/rest/security-advisories/repository-advisories#request-a-cve-for-a-repository-security-advisory>`_
        """
    self._requester.requestJsonAndCheck('POST', self.url + '/cve')