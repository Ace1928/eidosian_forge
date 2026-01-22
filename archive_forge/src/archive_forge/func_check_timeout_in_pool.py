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
def check_timeout_in_pool(self, world_type: str, agent_pool: List[AgentState], max_time_in_pool: int, backup_task: str=None):
    """
        Check for timed-out agents in pool.

        :param world_type:
            world type
        :param agent_pool:
            list of AgentStates
        :param max_time_in_pool:
            maximum time allowed for agent to be in pool
        :param backup_task:
            backup_task to start if we reach a timeout in the original pool
        """
    for agent_state in agent_pool:
        time_in_pool = agent_state.time_in_pool.get(world_type)
        if time_in_pool and time.time() - time_in_pool > max_time_in_pool:
            self.remove_agent_from_pool(agent_state, world_type)
            agent_state.set_active_agent(agent_state.get_overworld_agent())
            agent_state.stored_data['removed_after_timeout'] = True
            self.after_agent_removed(agent_state.service_id)
            if backup_task is not None:
                self.add_agent_to_pool(agent_state, backup_task)
            agent_state.stored_data['seen_wait_message'] = False
        elif time_in_pool and time.time() - time_in_pool > 30:
            if not agent_state.stored_data.get('seen_wait_message') or not agent_state.stored_data['seen_wait_message']:
                self.observe_message(agent_state.service_id, 'Pairing is taking longer than expected. If you wish to exit, type *EXIT*.')
                self.sender.typing_on(agent_state.service_id)
                agent_state.stored_data['seen_wait_message'] = True