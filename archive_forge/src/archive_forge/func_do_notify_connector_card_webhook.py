from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def do_notify_connector_card_webhook(module, webhook, payload):
    headers = {'Content-Type': 'application/json'}
    response, info = fetch_url(module=module, url=webhook, headers=headers, method='POST', data=payload)
    if info['status'] == 200:
        module.exit_json(changed=True)
    elif info['status'] == 400 and module.check_mode:
        if info['body'] == OFFICE_365_CARD_EMPTY_PAYLOAD_MSG:
            module.exit_json(changed=True)
        else:
            module.fail_json(msg=OFFICE_365_INVALID_WEBHOOK_MSG)
    else:
        module.fail_json(msg='failed to send %s as a connector card to Incoming Webhook: %s' % (payload, info['msg']))