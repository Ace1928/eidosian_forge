from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def run_zapi(self):
    """ calls the ZAPI """
    zapi_struct = self.zapi
    error = None
    if not isinstance(zapi_struct, dict):
        error = 'A directory entry is expected, eg: system-get-version: '
        zapi = zapi_struct
    else:
        zapi = list(zapi_struct.keys())
        if len(zapi) != 1:
            error = 'A single ZAPI can be called at a time'
        else:
            zapi = zapi[0]
    if error:
        self.module.fail_json(msg='%s, received: %s' % (error, zapi))
    zapi_obj = netapp_utils.zapi.NaElement(zapi)
    attributes = zapi_struct[zapi]
    if attributes is not None and attributes != 'None':
        zapi_obj.translate_struct(attributes)
    try:
        output = self.server.invoke_elem(zapi_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error running zapi %s: %s' % (zapi, to_native(error)), exception=traceback.format_exc())
    return self.jsonify_and_parse_output(output)