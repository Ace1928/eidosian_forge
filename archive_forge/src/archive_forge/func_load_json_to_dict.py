import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
def load_json_to_dict(json_path: Union[str, Path]) -> dict:
    """Load json file to a dictionary.

    Parameters:
        json_path (str): The path to the json file.

    Returns:
        (dict): The dictionary representation of the json file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data