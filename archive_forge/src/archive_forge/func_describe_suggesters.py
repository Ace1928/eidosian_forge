import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def describe_suggesters(self, domain_name, suggester_names=None, deployed=None):
    """
        Gets the suggesters configured for a domain. A suggester
        enables you to display possible matches before users finish
        typing their queries. Can be limited to specific suggesters by
        name. By default, shows all suggesters and includes any
        pending changes to the configuration. Set the `Deployed`
        option to `True` to show the active configuration and exclude
        pending changes. For more information, see `Getting Search
        Suggestions`_ in the Amazon CloudSearch Developer Guide .

        :type domain_name: string
        :param domain_name: The name of the domain you want to describe.

        :type suggester_names: list
        :param suggester_names: The suggesters you want to describe.

        :type deployed: boolean
        :param deployed: Whether to display the deployed configuration (
            `True`) or include any pending changes ( `False`). Defaults to
            `False`.

        """
    params = {'DomainName': domain_name}
    if suggester_names is not None:
        self.build_list_params(params, suggester_names, 'SuggesterNames.member')
    if deployed is not None:
        params['Deployed'] = str(deployed).lower()
    return self._make_request(action='DescribeSuggesters', verb='POST', path='/', params=params)