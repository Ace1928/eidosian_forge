import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def send_fb_payload(self, receiver_id, payload, quick_replies=None, persona_id=None):
    """
        Sends a payload to messenger, processes it if we can.
        """
    api_address = f'https://graph.facebook.com/{API_VERSION}/me/messages'
    if payload['type'] == 'list':
        data = create_compact_list_message(payload['data'])
    elif payload['type'] in ['image', 'video', 'file', 'audio']:
        data = create_attachment(payload)
    else:
        data = payload['data']
    message = {'messaging_type': 'RESPONSE', 'recipient': {'id': receiver_id}, 'message': {'attachment': data}}
    if quick_replies is not None:
        quick_replies = [create_reply_option(x, x) for x in quick_replies]
        message['message']['quick_replies'] = quick_replies
    if persona_id is not None:
        payload['persona_id'] = persona_id
    response = requests.post(api_address, params=self.auth_args, json=message)
    result = response.json()
    if 'error' in result:
        if result['error']['code'] == 1200:
            response = requests.post(api_address, params=self.auth_args, json=message)
            result = response.json()
    log_utils.print_and_log(logging.INFO, '"Facebook response from message send: {}"'.format(result))
    return result