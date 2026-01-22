from __future__ import (absolute_import, division, print_function)
import json
import time
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import apply_diff_key, job_tracking
def get_import_payload(module, rest_obj, view_id):
    attrib_dict = module.params.get('attributes').copy()
    import_payload = {}
    import_payload['Name'] = attrib_dict.pop('Name')
    template = get_template_by_name(import_payload['Name'], module, rest_obj)
    if template:
        module.exit_json(msg=TEMPLATE_NAME_EXISTS.format(name=import_payload['Name']))
    import_payload['ViewTypeId'] = view_id
    import_payload['Type'] = 2
    typeid = attrib_dict.get('Type') if attrib_dict.get('Type') else attrib_dict.get('TypeId')
    if typeid:
        if get_type_id_valid(rest_obj, typeid):
            import_payload['Type'] = typeid
        else:
            fail_module(module, msg="Type provided for 'import' operation is invalid")
    import_payload['Content'] = attrib_dict.pop('Content')
    if isinstance(attrib_dict, dict):
        attrib_dict.pop('TypeId', None)
        import_payload.update(attrib_dict)
    return import_payload