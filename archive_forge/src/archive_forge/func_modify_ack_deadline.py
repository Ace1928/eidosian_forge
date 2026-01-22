from concurrent import futures
import logging
from typing import NamedTuple, Callable
from google.cloud.pubsub_v1.subscriber.message import Message
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsub_v1.subscriber.exceptions import AcknowledgeStatus
def modify_ack_deadline(self, seconds: int):
    logging.warning('Likely incorrect call to modify_ack_deadline() on Pub/Sub Lite message. Pub/Sub Lite does not support redelivery in this way.')