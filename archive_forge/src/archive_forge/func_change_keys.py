from __future__ import absolute_import, division, print_function
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
def change_keys(recs, key='uuid', filter_func=None):
    """
    Take a xapi dict, and make the keys the value of recs[ref][key].

    Preserves the ref in rec['ref']

    """
    new_recs = {}
    for ref, rec in recs.items():
        if filter_func is not None and (not filter_func(rec)):
            continue
        for param_name, param_value in rec.items():
            if hasattr(param_value, 'value'):
                rec[param_name] = param_value.value
        new_recs[rec[key]] = rec
        new_recs[rec[key]]['ref'] = ref
    return new_recs