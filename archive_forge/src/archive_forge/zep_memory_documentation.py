from __future__ import annotations
from typing import Any, Dict, Optional
from langchain_community.chat_message_histories import ZepChatMessageHistory
from langchain.memory import ConversationBufferMemory
Save context from this conversation to buffer.

        Args:
            inputs (Dict[str, Any]): The inputs to the chain.
            outputs (Dict[str, str]): The outputs from the chain.
            metadata (Optional[Dict[str, Any]], optional): Any metadata to save with
                                                           the context. Defaults to None

        Returns:
            None
        