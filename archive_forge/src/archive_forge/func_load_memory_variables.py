from typing import Any, Dict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain.agents.format_scratchpad.openai_functions import (
from langchain.memory.chat_memory import BaseChatMemory
def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Return history buffer."""
    if self.return_messages:
        final_buffer: Any = self.buffer
    else:
        final_buffer = get_buffer_string(self.buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix)
    return {self.memory_key: final_buffer}