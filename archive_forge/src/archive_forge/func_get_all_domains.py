import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def get_all_domains(self, max_domains=None, next_token=None):
    """
        Returns a :py:class:`boto.resultset.ResultSet` containing
        all :py:class:`boto.sdb.domain.Domain` objects associated with
        this connection's Access Key ID.

        :keyword int max_domains: Limit the returned
            :py:class:`ResultSet <boto.resultset.ResultSet>` to the specified
            number of members.
        :keyword str next_token: A token string that was returned in an
            earlier call to this method as the ``next_token`` attribute
            on the returned :py:class:`ResultSet <boto.resultset.ResultSet>`
            object. This attribute is set if there are more than Domains than
            the value specified in the ``max_domains`` keyword. Pass the
            ``next_token`` value from you earlier query in this keyword to
            get the next 'page' of domains.
        """
    params = {}
    if max_domains:
        params['MaxNumberOfDomains'] = max_domains
    if next_token:
        params['NextToken'] = next_token
    return self.get_list('ListDomains', params, [('DomainName', Domain)])