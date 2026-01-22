from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def extract_volume(self):
    volume = self.get_volume()
    if not volume:
        self.module.fail_json(msg='Failed: volume not found')
    args = {'id': volume['id'], 'url': self.module.params.get('url'), 'mode': self.module.params.get('mode').upper(), 'zoneid': self.get_zone(key='id')}
    self.result['changed'] = True
    if not self.module.check_mode:
        res = self.query_api('extractVolume', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            volume = self.poll_job(res, 'volume')
        self.volume = volume
    return volume