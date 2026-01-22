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
def set_active_agent(self, active_agent: ChatServiceAgent):
    """
        Set active agent for this agent.

        :param active_agent:
            A ChatServiceAgent, the new active agent for this given agent state
        """
    self.active_agent = active_agent