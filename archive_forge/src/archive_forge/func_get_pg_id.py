from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.vexata import (
def get_pg_id(module, array):
    """Retrieve a named pg's id if it exists, error if absent."""
    name = module.params['pg']
    try:
        pgs = array.list_pgs()
        pg = filter(lambda pg: pg['name'] == name, pgs)
        if len(pg) == 1:
            return pg[0]['id']
        else:
            module.fail_json(msg='Port group {0} was not found.'.format(name))
    except Exception:
        module.fail_json(msg='Error while attempting to retrieve port groups.')