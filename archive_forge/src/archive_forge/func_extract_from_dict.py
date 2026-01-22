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
@classmethod
def extract_from_dict(cls, dictionary, left, right):
    result = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result[key] = cls.extract_from_dict(value, left, right)
        else:
            result[key] = value[left:right]
    return result