import asyncio
import logging
import pathlib
import queue
import tempfile
import threading
import wave
from enum import Enum
from typing import (
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import (
from langchain_core.runnables import RunnableConfig, RunnableSerializable
@property
def riva_pb2(self) -> 'riva.client.AudioEncoding':
    """Returns the Riva API object for the encoding."""
    riva_client = _import_riva_client()
    return getattr(riva_client.AudioEncoding, self)