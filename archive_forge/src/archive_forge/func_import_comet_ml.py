import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult
import langchain_community
from langchain_community.callbacks.utils import (
def import_comet_ml() -> Any:
    """Import comet_ml and raise an error if it is not installed."""
    try:
        import comet_ml
    except ImportError:
        raise ImportError('To use the comet_ml callback manager you need to have the `comet_ml` python package installed. Please install it with `pip install comet_ml`')
    return comet_ml