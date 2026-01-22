from collections import OrderedDict
import gymnasium as gym
from typing import Union, Dict, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
Ensure the unpacked state shapes match the DNC output