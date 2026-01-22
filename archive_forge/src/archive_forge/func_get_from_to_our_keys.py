import argparse
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from classy_vision.models.regnet import RegNet, RegNetParams
from huggingface_hub import cached_download, hf_hub_url
from torch import Tensor
from vissl.models.model_helpers import get_trunk_forward_outputs
from transformers import AutoImageProcessor, RegNetConfig, RegNetForImageClassification, RegNetModel
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
def get_from_to_our_keys(model_name: str) -> Dict[str, str]:
    """
    Returns a dictionary that maps from original model's key -> our implementation's keys
    """
    our_config = RegNetConfig(depths=[2, 7, 17, 1], hidden_sizes=[8, 8, 8, 8], groups_width=8)
    if 'in1k' in model_name:
        our_model = RegNetForImageClassification(our_config)
    else:
        our_model = RegNetModel(our_config)
    from_model = FakeRegNetVisslWrapper(RegNet(FakeRegNetParams(depth=27, group_width=1010, w_0=1744, w_a=620.83, w_m=2.52)))
    with torch.no_grad():
        from_model = from_model.eval()
        our_model = our_model.eval()
        x = torch.randn((1, 3, 32, 32))
        dest_tracker = Tracker(our_model)
        dest_traced = dest_tracker(x).parametrized
        pprint(dest_tracker.name2module)
        src_tracker = Tracker(from_model)
        src_traced = src_tracker(x).parametrized

    def to_params_dict(dict_with_modules):
        params_dict = OrderedDict()
        for name, module in dict_with_modules.items():
            for param_name, param in module.state_dict().items():
                params_dict[f'{name}.{param_name}'] = param
        return params_dict
    from_to_ours_keys = {}
    src_state_dict = to_params_dict(src_traced)
    dst_state_dict = to_params_dict(dest_traced)
    for (src_key, src_param), (dest_key, dest_param) in zip(src_state_dict.items(), dst_state_dict.items()):
        from_to_ours_keys[src_key] = dest_key
        logger.info(f'{src_key} -> {dest_key}')
    if 'in1k' in model_name:
        from_to_ours_keys['0.clf.0.weight'] = 'classifier.1.weight'
        from_to_ours_keys['0.clf.0.bias'] = 'classifier.1.bias'
    return from_to_ours_keys