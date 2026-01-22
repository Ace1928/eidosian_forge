from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def clone_multiple_from_api(self, records):
    """
        Process a list of record object (DNSRecord) after receiving from API.
        Return a list of modified clones of the records; the originals will not be modified.
        """
    return [self.clone_from_api(record) for record in records]