from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_occm_agents_by_name(self, rest_api, account_id, name, provider):
    """
        Collect a list of agents matching account_id, name, and provider.
        :return: list of agents, error
        """
    agents, error = self.get_occm_agents_by_account(rest_api, account_id)
    if isinstance(agents, dict) and 'agents' in agents:
        agents = [agent for agent in agents['agents'] if agent['name'] == name and agent['provider'] == provider]
    return (agents, error)