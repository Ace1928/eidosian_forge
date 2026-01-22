from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import \
from ansible.module_utils._text import to_native
def vg_probe(self, data):
    self.update_validation(data)
    params_mapping = (('ownershipgroup', data.get('owner_name', '')), ('ignoreuserfcmaps', data.get('ignore_user_flash_copy_maps', '')), ('replicationpolicy', data.get('replication_policy_name', '')), ('noownershipgroup', not bool(data.get('owner_name', ''))), ('nosafeguardpolicy', not bool(data.get('safeguarded_policy_name', ''))), ('nosnapshotpolicy', not bool(data.get('snapshot_policy_name', ''))), ('noreplicationpolicy', not bool(data.get('replication_policy_name', ''))))
    props = dict(((k, getattr(self, k)) for k, v in params_mapping if getattr(self, k) and getattr(self, k) != v))
    if self.safeguardpolicyname and self.safeguardpolicyname != data.get('safeguarded_policy_name', ''):
        props['safeguardedpolicy'] = self.safeguardpolicyname
        if self.policystarttime:
            props['policystarttime'] = self.policystarttime
    elif self.safeguardpolicyname:
        if self.policystarttime and self.policystarttime + '00' != data.get('safeguarded_policy_start_time', ''):
            props['safeguardedpolicy'] = self.safeguardpolicyname
            props['policystarttime'] = self.policystarttime
    elif self.snapshotpolicy and self.snapshotpolicy != data.get('snapshot_policy_name', ''):
        props['snapshotpolicy'] = self.snapshotpolicy
        props['safeguarded'] = self.safeguarded
        if self.policystarttime:
            props['policystarttime'] = self.policystarttime
    elif self.snapshotpolicy:
        if self.policystarttime and self.policystarttime + '00' != data.get('snapshot_policy_start_time', ''):
            props['snapshotpolicy'] = self.snapshotpolicy
            props['policystarttime'] = self.policystarttime
        if self.safeguarded not in ('', None) and self.safeguarded != strtobool(data.get('snapshot_policy_safeguarded', 0)):
            props['snapshotpolicy'] = self.snapshotpolicy
            props['safeguarded'] = self.safeguarded
    if self.snapshotpolicysuspended and self.snapshotpolicysuspended != data.get('snapshot_policy_suspended', ''):
        props['snapshotpolicysuspended'] = self.snapshotpolicysuspended
    self.log('volumegroup props = %s', props)
    return props