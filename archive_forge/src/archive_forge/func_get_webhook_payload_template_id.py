from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
def get_webhook_payload_template_id(meraki, templates, name):
    for template in templates:
        if template['name'] == name:
            return template['payloadTemplateId']
    meraki.fail_json(msg='No payload template found with the name {0}'.format(name))