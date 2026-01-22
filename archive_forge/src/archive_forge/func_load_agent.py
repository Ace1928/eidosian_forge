import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
def load_agent(self, load_dir=None):
    if load_dir is None:
        load_dir = self.model_dir
    self.pov_agent.load_agent(load_dir)