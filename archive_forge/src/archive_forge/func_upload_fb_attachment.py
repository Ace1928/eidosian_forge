import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def upload_fb_attachment(self, payload):
    """
        Uploads an attachment using the Attachment Upload API and returns an attachment
        ID.
        """
    api_address = f'https://graph.facebook.com/{API_VERSION}/me/message_attachments'
    assert payload['type'] in ['image', 'video', 'file', 'audio'], 'unsupported attachment type'
    if 'url' in payload:
        message = {'message': {'attachment': {'type': payload['type'], 'payload': {'is_reusable': 'true', 'url': payload['url']}}}}
        response = requests.post(api_address, params=self.auth_args, json=message)
    elif 'filename' in payload:
        message = {'attachment': {'type': payload['type'], 'payload': {'is_reusable': 'true'}}}
        with open(payload['filename'], 'rb') as f:
            filedata = {'filedata': (payload['filename'], f, payload['type'] + '/' + payload['format'])}
            response = requests.post(api_address, params=self.auth_args, data={'message': json.dumps(message)}, files=filedata)
    result = response.json()
    log_utils.print_and_log(logging.INFO, '"Facebook response from attachment upload: {}"'.format(result))
    return result