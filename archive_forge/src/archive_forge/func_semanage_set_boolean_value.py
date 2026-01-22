from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils._text import to_bytes, to_text
def semanage_set_boolean_value(module, handle, name, value):
    rc, t_b = semanage.semanage_bool_create(handle)
    if rc < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to create seboolean with semanage')
    if semanage.semanage_bool_set_name(handle, t_b, name) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to set seboolean name with semanage')
    rc, boolkey = semanage.semanage_bool_key_extract(handle, t_b)
    if rc < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to extract boolean key with semanage')
    rc, exists = semanage.semanage_bool_exists(handle, boolkey)
    if rc < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to check if boolean is defined')
    if not exists:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='SELinux boolean %s is not defined in persistent policy' % name)
    rc, sebool = semanage.semanage_bool_query(handle, boolkey)
    if rc < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to query boolean in persistent policy')
    semanage.semanage_bool_set_value(sebool, value)
    if semanage.semanage_bool_modify_local(handle, boolkey, sebool) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to modify boolean key with semanage')
    if semanage.semanage_bool_set_active(handle, boolkey, sebool) < 0:
        semanage.semanage_handle_destroy(handle)
        module.fail_json(msg='Failed to set boolean key active with semanage')
    semanage.semanage_bool_key_free(boolkey)
    semanage.semanage_bool_free(t_b)
    semanage.semanage_bool_free(sebool)