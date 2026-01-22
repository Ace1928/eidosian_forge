from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_cluster_snapshot_copy_status(self):
    response = self.client.describe_clusters(ClusterIdentifier=self.cluster_name)
    return response['Clusters'][0].get('ClusterSnapshotCopyStatus')