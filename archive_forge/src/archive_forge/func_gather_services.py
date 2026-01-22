from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
def gather_services(self):
    services = {}
    self.rcctl_path = self.module.get_bin_path('rcctl')
    if self.rcctl_path:
        for svc in self.query_rcctl('all'):
            services[svc] = {'name': svc, 'source': 'rcctl', 'rogue': False}
            services[svc].update(self.get_info(svc))
        for svc in self.query_rcctl('on'):
            services[svc].update({'status': 'enabled'})
        for svc in self.query_rcctl('started'):
            services[svc].update({'state': 'running'})
        for svc in self.query_rcctl('failed'):
            services[svc].update({'state': 'failed'})
        for svc in services.keys():
            if services[svc].get('status') is None:
                services[svc].update({'status': 'disabled'})
            if services[svc].get('state') is None:
                services[svc].update({'state': 'stopped'})
        for svc in self.query_rcctl('rogue'):
            services[svc]['rogue'] = True
    return services