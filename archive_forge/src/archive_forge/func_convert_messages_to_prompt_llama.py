from typing import List
from langchain_core.messages import (
def convert_messages_to_prompt_llama(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for llama."""
    return '\n'.join([_convert_one_message_to_text_llama(message) for message in messages])