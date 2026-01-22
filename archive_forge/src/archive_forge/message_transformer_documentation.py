from abc import ABC, abstractmethod
from typing import Callable
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite_v1 import SequencedMessage
Transform a SequencedMessage to a PubsubMessage.

        Args:
          source: The message to transform.

        Raises:
          GoogleAPICallError: To fail the client if raised inline.
        