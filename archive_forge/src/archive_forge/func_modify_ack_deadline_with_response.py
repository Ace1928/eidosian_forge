from concurrent import futures
import logging
from typing import NamedTuple, Callable
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
def modify_ack_deadline_with_response(self, seconds: int) -> 'futures.Future':
    logging.warning('Likely incorrect call to modify_ack_deadline_with_response() on Pub/Sub Lite message. Pub/Sub Lite does not support redelivery in this way.')
    return _SUCCESS_FUTURE