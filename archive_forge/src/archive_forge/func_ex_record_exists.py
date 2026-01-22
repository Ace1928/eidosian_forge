import json
from typing import Any, Dict, List, Optional
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import urlencode
from libcloud.common.vultr import (
def ex_record_exists(self, record_id, records_list):
    """
        :param record_id: Name of the `Record` object.
        :type record_id: ``str``

        :param records_list: A list containing `Record` objects
        :type records_list: ``list``

        :rtype: ``bool``
        """
    record_ids = []
    for record in records_list:
        record_ids.append(record.id)
    return record_id in record_ids