import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
def handle_agent_inputs(*args, **kwargs):
    args = [arg.to_raw() if isinstance(arg, AgentType) else arg for arg in args]
    kwargs = {k: v.to_raw() if isinstance(v, AgentType) else v for k, v in kwargs.items()}
    return (args, kwargs)