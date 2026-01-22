from __future__ import absolute_import
import copy
import logging
import os
import threading
import time
import typing
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union
import warnings
from google.api_core import gapic_v1
from google.auth.credentials import AnonymousCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.pubsub_v1 import types
from google.cloud.pubsub_v1.publisher import exceptions
from google.cloud.pubsub_v1.publisher import futures
from google.cloud.pubsub_v1.publisher._batch import thread
from google.cloud.pubsub_v1.publisher._sequencer import ordered_sequencer
from google.cloud.pubsub_v1.publisher._sequencer import unordered_sequencer
from google.cloud.pubsub_v1.publisher.flow_controller import FlowController
from google.pubsub_v1 import gapic_version as package_version
from google.pubsub_v1 import types as gapic_types
from google.pubsub_v1.services.publisher import client as publisher_client
def resume_publish(self, topic: str, ordering_key: str) -> None:
    """Resume publish on an ordering key that has had unrecoverable errors.

        Args:
            topic: The topic to publish messages to.
            ordering_key: A string that identifies related messages for which
                publish order should be respected.

        Raises:
            RuntimeError:
                If called after publisher has been stopped by a `stop()` method
                call.
            ValueError:
                If the topic/ordering key combination has not been seen before
                by this client.
        """
    with self._batch_lock:
        if self._is_stopped:
            raise RuntimeError('Cannot resume publish on a stopped publisher.')
        if not self._enable_message_ordering:
            raise ValueError('Cannot resume publish on a topic/ordering key if ordering is not enabled.')
        sequencer_key = (topic, ordering_key)
        sequencer = self._sequencers.get(sequencer_key)
        if sequencer is None:
            _LOGGER.debug('Error: The topic/ordering key combination has not been seen before.')
        else:
            sequencer.unpause()