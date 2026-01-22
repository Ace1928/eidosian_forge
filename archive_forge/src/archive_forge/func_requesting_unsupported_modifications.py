from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def requesting_unsupported_modifications(actual, requested):
    if actual['SnapshotCopyGrantName'] != requested['snapshot_copy_grant'] or actual['DestinationRegion'] != requested['destination_region']:
        return True
    return False