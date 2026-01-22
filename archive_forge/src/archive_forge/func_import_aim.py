from copy import deepcopy
from typing import Any, Dict, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
def import_aim() -> Any:
    """Import the aim python package and raise an error if it is not installed."""
    try:
        import aim
    except ImportError:
        raise ImportError('To use the Aim callback manager you need to have the `aim` python package installed.Please install it with `pip install aim`')
    return aim