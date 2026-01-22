from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def attrs_put(self, attrs):
    """Puts attributes on ECS container instance"""
    try:
        self.ecs.put_attributes(cluster=self.cluster, attributes=attrs.get_for_ecs_arn(self.ecs_arn))
    except ClientError as e:
        self.module.fail_json(msg=str(e))