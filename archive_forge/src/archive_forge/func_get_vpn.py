from __future__ import (absolute_import, division, print_function)
import time
def get_vpn(oneandone_conn, vpn, full_object=False):
    """
    Validates that the vpn exists by ID or a name.
    Returns the vpn if one was found.
    """
    for _vpn in oneandone_conn.list_vpns(per_page=1000):
        if vpn in (_vpn['id'], _vpn['name']):
            if full_object:
                return _vpn
            return _vpn['id']