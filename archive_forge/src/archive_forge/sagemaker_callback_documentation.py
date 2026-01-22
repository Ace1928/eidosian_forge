import json
import os
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Dict, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import (
Reset the steps and delete the temporary local directory.