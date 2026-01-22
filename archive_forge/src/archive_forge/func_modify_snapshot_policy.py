from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def modify_snapshot_policy(self, snap_pol_details, modify_dict):
    """
        Modify the snapshot policy attributes
        :param snap_pol_details: Details of the snapshot policy
        :param modify_dict: Dictionary containing the attributes of
         snapshot policy which are to be updated
        :return: True, if the operation is successful
        """
    try:
        msg = f'Dictionary containing attributes which are to be updated is {str(modify_dict)}.'
        LOG.info(msg)
        if not self.module.check_mode:
            if 'new_name' in modify_dict:
                self.powerflex_conn.snapshot_policy.rename(snap_pol_details['id'], modify_dict['new_name'])
                msg = f'The name of the volume is updated to {modify_dict['new_name']} sucessfully.'
                LOG.info(msg)
            if 'auto_snapshot_creation_cadence_in_min' in modify_dict and 'num_of_retained_snapshots_per_level' not in modify_dict:
                self.powerflex_conn.snapshot_policy.modify(snapshot_policy_id=snap_pol_details['id'], auto_snap_creation_cadence_in_min=modify_dict['auto_snapshot_creation_cadence_in_min'], retained_snaps_per_level=snap_pol_details['numOfRetainedSnapshotsPerLevel'])
                msg = f'The snapshot rule is updated to {modify_dict['auto_snapshot_creation_cadence_in_min']}'
                LOG.info(msg)
            elif 'auto_snapshot_creation_cadence_in_min' not in modify_dict and 'num_of_retained_snapshots_per_level' in modify_dict:
                self.powerflex_conn.snapshot_policy.modify(snapshot_policy_id=snap_pol_details['id'], auto_snap_creation_cadence_in_min=snap_pol_details['autoSnapshotCreationCadenceInMin'], retained_snaps_per_level=modify_dict['num_of_retained_snapshots_per_level'])
                msg = f'The retention rule is updated to {modify_dict['num_of_retained_snapshots_per_level']}'
                LOG.info(msg)
            elif 'auto_snapshot_creation_cadence_in_min' in modify_dict and 'num_of_retained_snapshots_per_level' in modify_dict:
                self.powerflex_conn.snapshot_policy.modify(snapshot_policy_id=snap_pol_details['id'], auto_snap_creation_cadence_in_min=modify_dict['auto_snapshot_creation_cadence_in_min'], retained_snaps_per_level=modify_dict['num_of_retained_snapshots_per_level'])
                msg = f'The snapshot rule is updated to {modify_dict['auto_snapshot_creation_cadence_in_min']} and the retention rule is updated to {modify_dict['num_of_retained_snapshots_per_level']}'
                LOG.info(msg)
        return True
    except Exception as e:
        err_msg = f'Failed to update the snapshot policy {snap_pol_details['id']} with error {str(e)}'
        LOG.error(err_msg)
        self.module.fail_json(msg=err_msg)