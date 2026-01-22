import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def get_timesteps(self, num_inference_steps, strength):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:].numpy()
    return (timesteps, num_inference_steps - t_start)