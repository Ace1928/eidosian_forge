from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
def convert_messages_for_finetuning(sessions: Iterable[ChatSession]) -> List[List[dict]]:
    """Convert messages to a list of lists of dictionaries for fine-tuning.

    Args:
        sessions: The chat sessions.

    Returns:
        The list of lists of dictionaries.
    """
    return [[convert_message_to_dict(s) for s in session['messages']] for session in sessions if _has_assistant_message(session)]