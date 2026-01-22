from __future__ import (absolute_import, division, print_function)
import time
def get_public_ip(oneandone_conn, public_ip, full_object=False):
    """
    Validates that the public ip exists by ID or a name.
    Returns the public ip if one was found.
    """
    for _public_ip in oneandone_conn.list_public_ips(per_page=1000):
        if public_ip in (_public_ip['id'], _public_ip['ip']):
            if full_object:
                return _public_ip
            return _public_ip['id']