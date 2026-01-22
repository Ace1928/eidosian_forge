from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def source_type_weight(self):
    """Give a weight on the type of this source.

        Basically make sure that IPv6Networks are sorted higher than IPv4Networks.
        This is a 'when all else fails' solution in __lt__.
        """
    if self['type'] == 'local':
        return 3
    sourceobj = self.source()
    if isinstance(sourceobj, ipaddress.IPv4Network):
        return 2
    if isinstance(sourceobj, ipaddress.IPv6Network):
        return 1
    if isinstance(sourceobj, str):
        return 0
    raise PgHbaValueError('This source {0} is of an unknown type...'.format(sourceobj))