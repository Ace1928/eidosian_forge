from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def full_config(self):
    if self._full_config_cache is None:
        self._full_config_cache = dict(QueueConfigurations=[], TopicConfigurations=[], LambdaFunctionConfigurations=[])
        try:
            config_lookup = self.client.get_bucket_notification_configuration(Bucket=self.bucket_name)
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json(msg=f'{e}')
        if config_lookup.get('QueueConfigurations'):
            for queue_config in config_lookup.get('QueueConfigurations'):
                self._full_config_cache['QueueConfigurations'].append(Config.from_api(queue_config))
        if config_lookup.get('TopicConfigurations'):
            for topic_config in config_lookup.get('TopicConfigurations'):
                self._full_config_cache['TopicConfigurations'].append(Config.from_api(topic_config))
        if config_lookup.get('LambdaFunctionConfigurations'):
            for function_config in config_lookup.get('LambdaFunctionConfigurations'):
                self._full_config_cache['LambdaFunctionConfigurations'].append(Config.from_api(function_config))
    return self._full_config_cache