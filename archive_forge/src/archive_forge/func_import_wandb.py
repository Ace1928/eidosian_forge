import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def import_wandb() -> Any:
    """Import the wandb python package and raise an error if it is not installed."""
    try:
        import wandb
    except ImportError:
        raise ImportError('To use the wandb callback manager you need to have the `wandb` python package installed. Please install it with `pip install wandb`')
    return wandb