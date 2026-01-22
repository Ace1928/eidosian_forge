from copy import deepcopy
from typing import Iterable, Iterator, List
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import AIMessage, BaseMessage
def merge_chat_runs(chat_sessions: Iterable[ChatSession]) -> Iterator[ChatSession]:
    """Merge chat runs together.

    A chat run is a sequence of messages from the same sender.

    Args:
        chat_sessions: A list of chat sessions.

    Returns:
        A list of chat sessions with merged chat runs.
    """
    for chat_session in chat_sessions:
        yield merge_chat_runs_in_session(chat_session)