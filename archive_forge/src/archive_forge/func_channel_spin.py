import copy
import functools
import logging
import os
import sys
import threading
import time
import types
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _grpcio_metadata  # pytype: disable=pyi-error
from grpc import _observability  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import IntegratedCallFactory
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import UserTag
import grpc.experimental  # pytype: disable=pyi-error
def channel_spin():
    while True:
        cygrpc.block_if_fork_in_progress(state)
        event = state.channel.next_call_event()
        if event.completion_type == cygrpc.CompletionType.queue_timeout:
            continue
        call_completed = event.tag(event)
        if call_completed:
            with state.lock:
                state.managed_calls -= 1
                if state.managed_calls == 0:
                    return