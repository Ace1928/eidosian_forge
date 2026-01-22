import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def batch_put_attributes(self, domain_or_name, items, replace=True):
    """
        Store attributes for multiple items in a domain.

        :type domain_or_name: string or :class:`boto.sdb.domain.Domain` object.
        :param domain_or_name: Either the name of a domain or a Domain object

        :type items: dict or dict-like object
        :param items: A dictionary-like object.  The keys of the dictionary are
                      the item names and the values are themselves dictionaries
                      of attribute names/values, exactly the same as the
                      attribute_names parameter of the scalar put_attributes
                      call.

        :type replace: bool
        :param replace: Whether the attribute values passed in will replace
                        existing values or will be added as addition values.
                        Defaults to True.

        :rtype: bool
        :return: True if successful
        """
    domain, domain_name = self.get_domain_and_name(domain_or_name)
    params = {'DomainName': domain_name}
    self._build_batch_list(params, items, replace)
    return self.get_status('BatchPutAttributes', params, verb='POST')