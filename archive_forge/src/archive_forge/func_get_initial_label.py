from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def get_initial_label(self) -> str:
    """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
    return f'{THINKING_EMOJI} **Thinking...**'