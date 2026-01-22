from typing import Any, Dict, List, Union
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain.memory.chat_memory import BaseChatMemory
@property
def buffer_as_str(self) -> str:
    """Exposes the buffer as a string in case return_messages is False."""
    messages = self.chat_memory.messages[-self.k * 2:] if self.k > 0 else []
    return get_buffer_string(messages, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)