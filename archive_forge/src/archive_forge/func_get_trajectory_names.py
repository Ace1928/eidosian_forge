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
@staticmethod
def get_trajectory_names(data_dir):
    result = [os.path.basename(x) for x in DataPipeline._get_all_valid_recordings(data_dir)]
    return sorted(result)