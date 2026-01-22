from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def create_snapcoll(client_obj, snapcoll_name, volcoll_name, **kwargs):
    if utils.is_null_or_empty(snapcoll_name):
        return (False, False, 'Create snapshot collection failed. snapshot collection name is not present.', {}, {})
    try:
        snapcoll_resp = client_obj.snapshot_collections.get(id=None, name=snapcoll_name, volcoll_name=volcoll_name)
        if utils.is_null_or_empty(snapcoll_resp):
            params = utils.remove_null_args(**kwargs)
            snapcoll_resp = client_obj.snapshot_collections.create(name=snapcoll_name, **params)
            return (True, True, f"Created snapshot collection '{snapcoll_name}' for volume collection '{volcoll_name}' successfully.", {}, snapcoll_resp.attrs)
        else:
            return (False, False, f"Snapshot collection '{snapcoll_name}' for volume collection '{volcoll_name}' cannot be createdas it is already present in given state.", {}, snapcoll_resp.attrs)
    except Exception as ex:
        return (False, False, f'Snapshot collection creation failed | {ex}', {}, {})