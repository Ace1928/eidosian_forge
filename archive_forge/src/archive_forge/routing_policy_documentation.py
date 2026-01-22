from abc import ABC, abstractmethod
from google.cloud.pubsublite.types.partition import Partition
from google.cloud.pubsublite_v1.types.common import PubSubMessage

        Route a message to a given partition.
        Args:
          message: The message to route

        Returns: The partition to route to

        