from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def get_ontap_version_from_params(self):
    """ Provide a way to override the current version
            This is required when running a custom vsadmin role as ONTAP does not currently allow access to /api/cluster.
            This may also be interesting for testing :)
            Report a warning if API call failed to report version.
            Report a warning if current version could be fetched and is different.
        """
    try:
        version = [int(x) for x in self.force_ontap_version.split('.')]
        if len(version) == 2:
            version.append(0)
        gen, major, minor = version
    except (TypeError, ValueError) as exc:
        self.module.fail_json(msg='Error: unexpected format in force_ontap_version, expecting G.M.m or G.M, as in 9.10.1, got: %s, error: %s' % (self.force_ontap_version, exc))
    warning = ''
    read_version = self.get_ontap_version()
    if read_version == (-1, -1, -1):
        warning = ', unable to read current version:'
    elif read_version != (gen, major, minor):
        warning = ' but current version is %s' % self.ontap_version['full']
    if warning:
        warning = 'Forcing ONTAP version to %s%s' % (self.force_ontap_version, warning)
        self.set_version({'version': {'generation': gen, 'major': major, 'minor': minor, 'full': 'set by user to %s' % self.force_ontap_version}})
    return warning