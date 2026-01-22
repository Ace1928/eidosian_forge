from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def handover_volcoll(client_obj, volcoll_name, **kwargs):
    if utils.is_null_or_empty(volcoll_name):
        return (False, False, 'Handover of volume collection failed as volume collection name is null.', {})
    try:
        volcoll_resp = client_obj.volume_collections.get(id=None, name=volcoll_name)
        params = utils.remove_null_args(**kwargs)
        if utils.is_null_or_empty(volcoll_resp):
            return (False, False, f"Volume collection '{volcoll_name}' not present for handover.", {})
        else:
            client_obj.volume_collections.handover(id=volcoll_resp.attrs.get('id'), **params)
            return (True, True, f"Handover of volume collection '{volcoll_name}' done successfully.", {})
    except Exception as ex:
        return (False, False, f'Handover of volume collection failed | {ex}', {})