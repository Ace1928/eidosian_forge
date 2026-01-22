import contextlib
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from ..core import randn_tensor
from ..import_utils import is_peft_available
from .sd_utils import convert_state_dict_to_diffusers
def get_trainable_layers(self):
    if self.use_lora:
        lora_config = LoraConfig(r=4, lora_alpha=4, init_lora_weights='gaussian', target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'])
        self.sd_pipeline.unet.add_adapter(lora_config)
        for param in self.sd_pipeline.unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        return self.sd_pipeline.unet
    else:
        return self.sd_pipeline.unet