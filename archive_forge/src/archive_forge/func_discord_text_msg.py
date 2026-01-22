from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import AnsibleModule
def discord_text_msg(module):
    webhook_id = module.params['webhook_id']
    webhook_token = module.params['webhook_token']
    content = module.params['content']
    user = module.params['username']
    avatar_url = module.params['avatar_url']
    tts = module.params['tts']
    embeds = module.params['embeds']
    headers = {'content-type': 'application/json'}
    url = 'https://discord.com/api/webhooks/%s/%s' % (webhook_id, webhook_token)
    payload = {'content': content, 'username': user, 'avatar_url': avatar_url, 'tts': tts, 'embeds': embeds}
    payload = module.jsonify(payload)
    response, info = fetch_url(module, url, data=payload, headers=headers, method='POST')
    return (response, info)