import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def define_suggester(self, domain_name, suggester):
    """
        Configures a suggester for a domain. A suggester enables you
        to display possible matches before users finish typing their
        queries. When you configure a suggester, you must specify the
        name of the text field you want to search for possible matches
        and a unique name for the suggester. For more information, see
        `Getting Search Suggestions`_ in the Amazon CloudSearch
        Developer Guide .

        :type domain_name: string
        :param domain_name: A string that represents the name of a domain.
            Domain names are unique across the domains owned by an account
            within an AWS region. Domain names start with a letter or number
            and can contain the following characters: a-z (lowercase), 0-9, and
            - (hyphen).

        :type suggester: dict
        :param suggester: Configuration information for a search suggester.
            Each suggester has a unique name and specifies the text field you
            want to use for suggestions. The following options can be
            configured for a suggester: `FuzzyMatching`, `SortExpression`.

        """
    params = {'DomainName': domain_name}
    self.build_complex_param(params, 'Suggester', suggester)
    return self._make_request(action='DefineSuggester', verb='POST', path='/', params=params)