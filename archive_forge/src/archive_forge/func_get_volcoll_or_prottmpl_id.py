from __future__ import absolute_import, division, print_function
import datetime
import uuid
def get_volcoll_or_prottmpl_id(client_obj, volcoll_name, prot_template_name):
    if is_null_or_empty(volcoll_name) and is_null_or_empty(prot_template_name):
        return None
    if is_null_or_empty(volcoll_name) is False and is_null_or_empty(prot_template_name) is False:
        raise Exception('Volcoll and prot_template are mutually exlusive. Please provide either one of them.')
    else:
        if volcoll_name is not None:
            resp = get_volcoll_id(client_obj, volcoll_name)
            if resp is None:
                raise Exception(f'Invalid value for volcoll: {volcoll_name}')
        elif prot_template_name is not None:
            resp = get_prottmpl_id(client_obj, prot_template_name)
            if resp is None:
                raise Exception(f'Invalid value for protection template {prot_template_name}')
        return resp