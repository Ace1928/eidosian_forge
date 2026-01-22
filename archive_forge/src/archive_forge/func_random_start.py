from pettingzoo import AECEnv
from pettingzoo.classic.chess.chess import raw_env as chess_v5
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Any
import chess as ch
import numpy as np
def random_start(self, random_moves):
    self.env.board = ch.Board()
    for i in range(random_moves):
        self.env.board.push(np.random.choice(list(self.env.board.legal_moves)))
    return self.env.board