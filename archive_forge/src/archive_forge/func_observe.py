from pettingzoo import AECEnv
from pettingzoo.classic.chess.chess import raw_env as chess_v5
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Any
import chess as ch
import numpy as np
def observe(self):
    return {self.env.agent_selection: self.env.observe(self.env.agent_selection), 'state': self.get_state()}