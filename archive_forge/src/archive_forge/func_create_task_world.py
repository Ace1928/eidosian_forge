import copy
import random
from typing import List, Dict, Union
from parlai.core.agents import create_agents_from_shared
from parlai.core.loader import load_task_module, load_world_module
from parlai.core.metrics import aggregate_named_reports
from parlai.core.opt import Opt
from parlai.core.teachers import Teacher, create_task_agent_from_taskname
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import Timer, display_messages
from parlai.tasks.tasks import ids_to_tasks
import parlai.utils.logging as logging
def create_task_world(opt: Opt, user_agents, default_world=None):
    """
    Instantiate a world with the supplied options and user agents.

    (A world factory.)
    """
    task_agents = _create_task_agents(opt)
    world_class = load_world_module(opt['task'], interactive_task=opt.get('interactive_task', False), selfchat_task=opt.get('selfchat_task', False), num_agents=len(user_agents + task_agents), default_world=default_world)
    return world_class(opt, task_agents + user_agents)