from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_prottmpl_id(client_obj, prottmpl_name):
    if is_null_or_empty(prottmpl_name):
        return None
    else:
        resp = client_obj.protection_templates.get(name=prottmpl_name)
        if resp is None:
            raise Exception(f'Invalid value for protection template {prottmpl_name}')
        return resp.attrs.get('id')