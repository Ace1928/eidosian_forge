from copy import deepcopy
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union
import numpy as np
from gym import Env
from gym.spaces import Space
from gym.vector.utils import concatenate, create_empty_array, iterate
from gym.vector.vector_env import VectorEnv
Close the environments.