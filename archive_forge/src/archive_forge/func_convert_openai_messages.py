from __future__ import annotations
import importlib
from typing import (
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import (
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import Literal
def convert_openai_messages(messages: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    """Convert dictionaries representing OpenAI messages to LangChain format.

    Args:
        messages: List of dictionaries representing OpenAI messages

    Returns:
        List of LangChain BaseMessage objects.
    """
    return [convert_dict_to_message(m) for m in messages]