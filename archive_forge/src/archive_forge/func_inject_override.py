from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.utils.misc import AttrDict
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils
from worlds import WizardEval, TopicsGenerator, TopicChooseWorld
from task_config import task_config
from projects.wizard_of_wikipedia.knowledge_retriever.knowledge_retriever import (
import gc
import datetime
import json
import os
import sys
from parlai.utils.logging import ParlaiLogger, INFO
def inject_override(opt, override_dict):
    opt['override'] = override_dict
    for k, v in override_dict.items():
        opt[k] = v