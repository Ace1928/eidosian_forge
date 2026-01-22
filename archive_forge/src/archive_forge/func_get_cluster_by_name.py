from __future__ import (absolute_import, division, print_function)
import time
import ssl
from os import environ
from ansible.module_utils.six import string_types
from ansible.module_utils.basic import AnsibleModule
def get_cluster_by_name(self, name):
    """
        Returns a cluster given its name.
        Args:
            name: the name of the cluster

        Returns: the cluster object or None if the host is absent.
        """
    clusters = self.one.clusterpool.info()
    for c in clusters.CLUSTER:
        if c.NAME == name:
            return c
    return None