import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
def format_prompt(self, task, chat_mode=False):
    description = '\n'.join([f'- {name}: {tool.description}' for name, tool in self.toolbox.items()])
    if chat_mode:
        if self.chat_history is None:
            prompt = self.chat_prompt_template.replace('<<all_tools>>', description)
        else:
            prompt = self.chat_history
        prompt += CHAT_MESSAGE_PROMPT.replace('<<task>>', task)
    else:
        prompt = self.run_prompt_template.replace('<<all_tools>>', description)
        prompt = prompt.replace('<<prompt>>', task)
    return prompt