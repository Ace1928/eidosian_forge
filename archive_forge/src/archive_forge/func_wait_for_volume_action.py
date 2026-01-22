from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def wait_for_volume_action(self, timeout=None):
    """Waits until volume action is complete is complete.
        :param: int timeout: Wait duration measured in seconds. Waits indefinitely when None.
        """
    action = 'unknown'
    percent_complete = None
    while action != 'complete':
        sleep(5)
        try:
            rc, operations = self.request('storage-systems/%s/symbol/getLongLivedOpsProgress' % self.ssid)
            action = 'complete'
            for operation in operations['longLivedOpsProgress']:
                if operation['volAction'] is not None:
                    for key in operation.keys():
                        if operation[key] is not None and 'volumeRef' in operation[key] and (operation[key]['volumeRef'] == self.volume_detail['id'] or ('storageVolumeRef' in self.volume_detail and operation[key]['volumeRef'] == self.volume_detail['storageVolumeRef'])):
                            action = operation['volAction']
                            percent_complete = operation['init']['percentComplete']
        except Exception as err:
            self.module.fail_json(msg='Failed to get volume expansion progress. Volume [%s]. Array Id [%s]. Error[%s].' % (self.name, self.ssid, to_native(err)))
        if timeout is not None:
            if timeout <= 0:
                self.module.warn('Expansion action, %s, failed to complete during the allotted time. Time remaining [%s]. Array Id [%s].' % (action, percent_complete, self.ssid))
                self.module.fail_json(msg='Expansion action failed to complete. Time remaining [%s]. Array Id [%s].' % (percent_complete, self.ssid))
            if timeout:
                timeout -= 5
        self.module.log('Expansion action, %s, is %s complete.' % (action, percent_complete))
    self.module.log('Expansion action is complete.')