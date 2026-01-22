from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def does_state_need_changing(attribute_name, desired_state, current_spec):
    """Run checks to see if the table needs to be modified. Basically a dirty check."""
    if not current_spec:
        return True
    if desired_state.lower() == 'enable' and current_spec.get('TimeToLiveStatus') not in ['ENABLING', 'ENABLED']:
        return True
    if desired_state.lower() == 'disable' and current_spec.get('TimeToLiveStatus') not in ['DISABLING', 'DISABLED']:
        return True
    if attribute_name != current_spec.get('AttributeName'):
        return True
    return False