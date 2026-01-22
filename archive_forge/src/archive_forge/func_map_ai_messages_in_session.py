from copy import deepcopy
from typing import Iterable, Iterator, List
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage
def map_ai_messages_in_session(chat_sessions: ChatSession, sender: str) -> ChatSession:
    """Convert messages from the specified 'sender' to AI messages.

    This is useful for fine-tuning the AI to adapt to your voice.
    """
    messages = []
    num_converted = 0
    for message in chat_sessions['messages']:
        if message.additional_kwargs.get('sender') == sender:
            message = AIMessage(content=message.content, additional_kwargs=message.additional_kwargs.copy(), example=getattr(message, 'example', None))
            num_converted += 1
        messages.append(message)
    return ChatSession(messages=messages)