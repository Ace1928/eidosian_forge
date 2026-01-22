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
def get_agent_state(self, agent_id: int) -> Optional[AgentState]:
    """
        Return agent state.

        :param agent_id:
            agent identifier

        :return:
            AgentState object if agent_id is being tracked, else None
        """
    if agent_id in self.messenger_agent_states:
        return self.messenger_agent_states[agent_id]
    return None