from __future__ import absolute_import, division, print_function
import time
def force_reboot_rediscache(self):
    """
        Force reboot specified redis cache instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Force reboot the redis cache instance {0}'.format(self.name))
    try:
        params = RedisRebootParameters(reboot_type=self.reboot['reboot_type'], shard_id=self.reboot.get('shard_id'))
        response = self._client.redis.force_reboot(resource_group_name=self.resource_group, name=self.name, parameters=params)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
        if self.wait_for_provisioning:
            self.wait_for_redis_running()
    except Exception as e:
        self.log('Error attempting to force reboot the redis cache instance.')
        self.fail('Error force rebooting the redis cache instance: {0}'.format(str(e)))
    return True