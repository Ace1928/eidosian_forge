from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import MerakiModule, meraki_argument_spec
def lookup_application(meraki, net_id, application):
    response = get_applications(meraki, net_id)
    for category in response['applicationCategories']:
        if category['name'].lower() == application.lower():
            return category['id']
        for app in category['applications']:
            if app['name'].lower() == application.lower():
                return app['id']
    meraki.fail_json(msg='No application or category named {0} found'.format(application))