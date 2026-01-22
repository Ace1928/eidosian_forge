import copy
from typing import List, Tuple, Optional, TypeVar
from parlai.core.agents import Agent, create_agent_from_shared
from parlai.core.image_featurizers import ImageLoader
from parlai.core.loader import load_teacher_module
from parlai.core.loader import register_teacher  # noqa: F401
from parlai.core.message import Message
from parlai.core.metrics import TeacherMetrics, aggregate_named_reports
from parlai.core.opt import Opt
from parlai.utils.conversations import Conversations
from parlai.utils.data import DatatypeHelper
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once
from parlai.utils.distributed import get_rank, num_workers, is_distributed
import parlai.utils.logging as logging
from abc import ABC, abstractmethod
import concurrent.futures
from threading import Thread
import queue
import random
import time
import os
import torch
import json
import argparse
def create_task_agent_from_taskname(opt: Opt):
    """
    Create task agent(s) assuming the input ``task_dir:teacher_class``.

    e.g. def_string is a shorthand path like ``babi:Task1k:1`` or ``#babi`` or a
    complete path like ``parlai.tasks.babi.agents:Task1kTeacher:1``, which essentially
    performs ``from parlai.tasks.babi import Task1kTeacher`` with the parameter ``1`` in
    ``opt['task']`` to be used by the class ``Task1kTeacher``.
    """
    if not opt.get('task'):
        raise RuntimeError('No task specified. Please select a task with ' + '--task {task_name}.')
    if ',' not in opt['task']:
        teacher_class = load_teacher_module(opt['task'])
        _add_task_flags_to_agent_opt(teacher_class, opt, opt['task'])
        task_agents = teacher_class(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents
    else:
        task_agents = MultiTaskTeacher(opt)
        if type(task_agents) != list:
            task_agents = [task_agents]
        return task_agents