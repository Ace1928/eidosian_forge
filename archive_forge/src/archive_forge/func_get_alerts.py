from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def get_alerts(self, alert_descriptions):
    """ Get a list of alert hrefs from a list of alert descriptions
        """
    alerts = []
    for alert_description in alert_descriptions:
        alert = self.manageiq.find_collection_resource_or_fail('alert_definitions', description=alert_description)
        alerts.append(alert['href'])
    return alerts