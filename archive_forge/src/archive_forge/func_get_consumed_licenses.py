import urllib.parse
from typing import Any, Dict
from github.EnterpriseConsumedLicenses import EnterpriseConsumedLicenses
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
def get_consumed_licenses(self) -> EnterpriseConsumedLicenses:
    """
        :calls: `GET /enterprises/{enterprise}/consumed-licenses <https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses>`_
        """
    headers, data = self._requester.requestJsonAndCheck('GET', self.url + '/consumed-licenses')
    if 'url' not in data:
        data['url'] = self.url + '/consumed-licenses'
    return EnterpriseConsumedLicenses(self._requester, headers, data, completed=True)