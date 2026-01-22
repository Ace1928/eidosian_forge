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
def observe_message(self, socket_id, message, quick_replies=None):
    """
        Send a message through the message manager.

        :param socket_id:
            int identifier for agent socket to send message to
        :param message:
            (str) message to send through the socket.
        :param quick_replies:
            (list) list of strings to send as quick replies.

        Returns a tornado future for tracking the `write_message` action.
        """
    if quick_replies is not None:
        quick_replies = list(quick_replies)
    message = json.dumps({'text': message.replace('\n', '<br />'), 'quick_replies': quick_replies})
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if socket_id not in self.subs:
        self.agent_id_to_overworld_future[socket_id].cancel()
        return
    return loop.run_until_complete(self.subs[socket_id].write_message(message))