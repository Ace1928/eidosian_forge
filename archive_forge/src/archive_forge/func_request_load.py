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
def request_load(self, receive_fn, load_fn, args):
    """
        Queue a request for loading.

        :param receive_fn:
            a receive function (for receiving the data)
        :param load_fn:
            a load function (for loading the data)
        :param args:
            arguments for the load function. args can be either a dictionary of
            arguments for a function, or a list of positional arguments
        """
    self.request_queue.put((receive_fn, load_fn, args))