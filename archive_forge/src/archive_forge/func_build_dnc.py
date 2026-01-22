from collections import OrderedDict
import gymnasium as gym
from typing import Union, Dict, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def build_dnc(self, device_idx: Union[int, None]) -> None:
    self.dnc = self.cfg['dnc_model'](input_size=self.cfg['preprocessor_output_size'], hidden_size=self.cfg['hidden_size'], num_layers=self.cfg['num_layers'], num_hidden_layers=self.cfg['num_hidden_layers'], read_heads=self.cfg['read_heads'], cell_size=self.cfg['cell_size'], nr_cells=self.cfg['nr_cells'], nonlinearity=self.cfg['nonlinearity'], gpu_id=device_idx)