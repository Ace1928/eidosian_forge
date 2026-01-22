from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
def requires_sg_version(self, module_or_option, version):
    return '%s requires StorageGRID %s or later.' % (module_or_option, version)