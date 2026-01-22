from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.urls import build_service_uri
from ..module_utils.teem import send_teem
def normalize_tables(self, tables):
    result = []
    for table in tables:
        tmp = dict()
        name = table.get('name', None)
        if name is None:
            raise F5ModuleError('One of the provided tables does not have a name')
        tmp['name'] = str(name)
        columns = table.get('columnNames', None)
        if columns:
            tmp['columnNames'] = [str(x) for x in columns]
            rows = table.get('rows', None)
            if rows:
                tmp['rows'] = []
                for row in rows:
                    tmp['rows'].append(dict(row=[str(x) for x in row['row']]))
        result.append(tmp)
    result = sorted(result, key=lambda k: k['name'])
    return result