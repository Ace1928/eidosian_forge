import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def send_sender_action(self, receiver_id, action, persona_id=None):
    api_address = f'https://graph.facebook.com/{API_VERSION}/me/messages'
    message = {'recipient': {'id': receiver_id}, 'sender_action': action}
    if persona_id is not None:
        message['persona_id'] = persona_id
    requests.post(api_address, params=self.auth_args, json=message)