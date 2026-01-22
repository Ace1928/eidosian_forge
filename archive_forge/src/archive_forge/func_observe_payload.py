import json
import asyncio
import logging
from parlai.core.agents import create_agent
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
from parlai.chat_service.services.websocket.sockets import MessageSocketHandler
from agents import WebsocketAgent
import tornado
from tornado.options import options
def observe_payload(self, socket_id, payload, quick_replies=None):
    """
        Send a message through the message manager.

        :param socket_id:
            int identifier for agent socket to send message to
        :param payload:
            (dict) payload to send through the socket. The mandatory keys are:
                    'type': (str) Type of the payload (e.g. 'image')
                    'data': str. base64 encoded content
                    If 'type' is 'image', the 'mime_type' (str) key can be provided
                    to specify the Mime type of the image

        Returns a tornado future for tracking the `write_message` action.
        """
    message = {'text': '', 'payload': payload, 'quick_replies': quick_replies}
    payload = json.dumps(message)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if socket_id not in self.subs:
        self.agent_id_to_overworld_future[socket_id].cancel()
        return
    return loop.run_until_complete(self.subs[socket_id].write_message(message))