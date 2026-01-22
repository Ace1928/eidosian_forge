from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_volume_property_changes(self):
    """Retrieve the volume update request body when change(s) are required.

        :raise AnsibleFailJson when attempting to change segment size on existing volume.
        :return dict: request body when change(s) to a volume's properties are required.
        """
    change = False
    request_body = dict(flashCache=self.ssd_cache_enabled, metaTags=[], cacheSettings=dict(readCacheEnable=self.read_cache_enable, writeCacheEnable=self.write_cache_enable))
    if self.segment_size_kb * 1024 != int(self.volume_detail['segmentSize']):
        self.module.fail_json(msg='Existing volume segment size is %s and cannot be modified.' % self.volume_detail['segmentSize'])
    if self.read_cache_enable != self.volume_detail['cacheSettings']['readCacheEnable'] or self.write_cache_enable != self.volume_detail['cacheSettings']['writeCacheEnable'] or self.ssd_cache_enabled != self.volume_detail['flashCached']:
        change = True
    if self.owning_controller_id and self.owning_controller_id != self.volume_detail['preferredManager']:
        change = True
        request_body.update(dict(owningControllerId=self.owning_controller_id))
    if self.workload_name:
        request_body.update(dict(metaTags=[dict(key='workloadId', value=self.workload_id), dict(key='volumeTypeId', value='volume')]))
        if {'key': 'workloadId', 'value': self.workload_id} not in self.volume_detail['metadata']:
            change = True
    elif self.volume_detail['metadata']:
        change = True
    if self.thin_provision:
        if self.thin_volume_growth_alert_threshold != int(self.volume_detail['growthAlertThreshold']):
            change = True
            request_body.update(dict(growthAlertThreshold=self.thin_volume_growth_alert_threshold))
        if self.thin_volume_expansion_policy != self.volume_detail['expansionPolicy']:
            change = True
            request_body.update(dict(expansionPolicy=self.thin_volume_expansion_policy))
    else:
        if self.read_ahead_enable != (int(self.volume_detail['cacheSettings']['readAheadMultiplier']) > 0):
            change = True
            request_body['cacheSettings'].update(dict(readAheadEnable=self.read_ahead_enable))
        if self.cache_without_batteries != self.volume_detail['cacheSettings']['cwob']:
            change = True
            request_body['cacheSettings'].update(dict(cacheWithoutBatteries=self.cache_without_batteries))
    return request_body if change else dict()