from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def build_payload_for_rocketchat(module, text, channel, username, icon_url, icon_emoji, link_names, color, attachments):
    payload = {}
    if color == 'normal' and text is not None:
        payload = dict(text=text)
    elif text is not None:
        payload = dict(attachments=[dict(text=text, color=color)])
    if channel is not None:
        if channel[0] == '#' or channel[0] == '@':
            payload['channel'] = channel
        else:
            payload['channel'] = '#' + channel
    if username is not None:
        payload['username'] = username
    if icon_emoji is not None:
        payload['icon_emoji'] = icon_emoji
    else:
        payload['icon_url'] = icon_url
    if link_names is not None:
        payload['link_names'] = link_names
    if attachments is not None:
        if 'attachments' not in payload:
            payload['attachments'] = []
    if attachments is not None:
        for attachment in attachments:
            if 'fallback' not in attachment:
                attachment['fallback'] = attachment['text']
            payload['attachments'].append(attachment)
    payload = 'payload=' + module.jsonify(payload)
    return payload