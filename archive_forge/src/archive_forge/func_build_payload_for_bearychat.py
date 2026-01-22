from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def build_payload_for_bearychat(module, text, markdown, channel, attachments):
    payload = {}
    if text is not None:
        payload['text'] = text
    if markdown is not None:
        payload['markdown'] = markdown
    if channel is not None:
        payload['channel'] = channel
    if attachments is not None:
        payload.setdefault('attachments', []).extend((build_payload_for_bearychat_attachment(module, item.get('title'), item.get('text'), item.get('color'), item.get('images')) for item in attachments))
    payload = 'payload=%s' % module.jsonify(payload)
    return payload