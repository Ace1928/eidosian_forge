from __future__ import (absolute_import, division, print_function)
import time
def get_appliance(oneandone_conn, appliance, full_object=False):
    """
    Validates the appliance exists by ID or name.
    Return the appliance ID.
    """
    for _appliance in oneandone_conn.list_appliances(q='IMAGE'):
        if appliance in (_appliance['id'], _appliance['name']):
            if full_object:
                return _appliance
            return _appliance['id']