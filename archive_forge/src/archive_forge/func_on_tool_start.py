from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.streamlit.mutable_expander import MutableExpander
def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
    self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
    self._prune_old_thought_containers()