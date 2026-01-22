from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_matching_filesystem(self, criteria):
    if criteria['device'] is not None:
        criteria['device'] = os.path.realpath(criteria['device'])
    self.__check_init()
    matching = [f for f in self.__filesystems.values() if self.__filesystem_matches_criteria(f, criteria)]
    if len(matching) == 1:
        return matching[0]
    else:
        raise BtrfsModuleException('Found %d filesystems matching criteria uuid=%s label=%s device=%s' % (len(matching), criteria['uuid'], criteria['label'], criteria['device']))