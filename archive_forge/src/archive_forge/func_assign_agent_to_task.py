from abc import ABC, abstractmethod
from asyncio import Future
import copy
import sys
import logging
import datetime
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from parlai.chat_service.core.agents import ChatServiceAgent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.core.world_runner import ChatServiceWorldRunner
from parlai.core.opt import Opt
def assign_agent_to_task(self, agent: ChatServiceAgent, task_id: str):
    """
        Mark agent in task.

        :param agent:
            ChatServiceAgent object to mark in task
        :param task_id:
            string task name
        """
    self.task_id_to_agent[task_id] = agent