from copy import deepcopy
from typing import Any, Dict, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
@property
def ignore_retriever(self) -> bool:
    """Whether to ignore retriever callbacks."""
    return self.ignore_retriever_