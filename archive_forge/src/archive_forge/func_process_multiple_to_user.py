from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def process_multiple_to_user(self, records):
    """
        Process a list of record objects (DNSRecord) for sending to the user.
        Modifies the records in-place.
        """
    for record in records:
        self.process_to_user(record)
    return records