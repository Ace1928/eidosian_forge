import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_query(self, type, filter=None, page=1, page_size=100, sort_asc=None, sort_desc=None):
    """
        Queries vCloud for specified type. See
        http://www.vmware.com/pdf/vcd_15_api_guide.pdf for details. Each
        element of the returned list is a dictionary with all attributes from
        the record.

        :param type: type to query (r.g. user, group, vApp etc.)
        :type  type: ``str``

        :param filter: filter expression (see documentation for syntax)
        :type  filter: ``str``

        :param page: page number
        :type  page: ``int``

        :param page_size: page size
        :type  page_size: ``int``

        :param sort_asc: sort in ascending order by specified field
        :type  sort_asc: ``str``

        :param sort_desc: sort in descending order by specified field
        :type  sort_desc: ``str``

        :rtype: ``list`` of dict
        """
    params = {'type': type, 'pageSize': page_size, 'page': page}
    if sort_asc:
        params['sortAsc'] = sort_asc
    if sort_desc:
        params['sortDesc'] = sort_desc
    url = '/api/query?' + urlencode(params)
    if filter:
        if not filter.startswith('('):
            filter = '(' + filter + ')'
        url += '&filter=' + filter.replace(' ', '+')
    results = []
    res = self.connection.request(url)
    for elem in res.object:
        if not elem.tag.endswith('Link'):
            result = elem.attrib
            result['type'] = elem.tag.split('}')[1]
            results.append(result)
    return results