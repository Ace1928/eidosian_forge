from parlai import __file__ as parlai_filepath
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.acute_eval.run import AcuteEvaluator, add_args as acute_add_args
from parlai.scripts.self_chat import self_chat, setup_args as self_chat_setup_args
from parlai.utils.conversations import Conversations, Conversation
from parlai.utils.strings import normalize_reply
from parlai.utils.testing import capture_output
from parlai.mturk.tasks.acute_eval.analysis import (
from parlai.mturk.tasks.acute_eval.dump_task_to_acute_format import (
from parlai.mturk.tasks.acute_eval.configs import CONFIG
from typing import Dict, Any, List, Tuple, Set
from itertools import combinations
import datetime
import time
import json
import os
import random
import torch
import hashlib
def compile_chat_logs(self):
    """
        Compile Chat Logs.

        Logs are generated depending on what is specified in the config for the model:
        1. If a `model` is provided, run selfchat for model
        2. If a `log_path` is provided, simply load the log path
        3. If a `task` is provided, convert the task to ACUTE format and load that.
        """
    for model in self.config_ids:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        outfile = self._get_log_path(model)
        if not os.path.exists(outfile):
            if 'model' in CONFIG[model]:
                self._run_selfchat(model)
            elif 'task' in CONFIG[model]:
                self._convert_task_to_conversations(model)
            else:
                raise RuntimeError(f'Path must exist if log_path specified for {model}')
            if os.path.exists(outfile):
                self._print_progress(f'Chats saved to {outfile} for {model}')
        self._print_progress(f'Chats already exist in {outfile}, moving on...')
        self.chat_files[model] = outfile