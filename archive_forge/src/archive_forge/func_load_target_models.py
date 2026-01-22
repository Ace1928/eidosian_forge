from typing import Optional, Union
import torch
import os
import numpy as np
import torchaudio
import warnings
from pathlib import Path
from contextlib import redirect_stderr
import io
import json
import openunmix
from openunmix import model
def load_target_models(targets, model_str_or_path='umxl', device='cpu', pretrained=True):
    """Core model loader

    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)

    The loader either loads the models from a known model string
    as registered in the __init__.py or loads from custom configs.
    """
    if isinstance(targets, str):
        targets = [targets]
    model_path = Path(model_str_or_path).expanduser()
    if not model_path.exists():
        try:
            hub_loader = getattr(openunmix, model_str_or_path + '_spec')
            err = io.StringIO()
            with redirect_stderr(err):
                return hub_loader(targets=targets, device=device, pretrained=pretrained)
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
    else:
        models = {}
        for target in targets:
            with open(Path(model_path, target + '.json'), 'r') as stream:
                results = json.load(stream)
            target_model_path = next(Path(model_path).glob('%s*.pth' % target))
            state = torch.load(target_model_path, map_location=device)
            models[target] = model.OpenUnmix(nb_bins=results['args']['nfft'] // 2 + 1, nb_channels=results['args']['nb_channels'], hidden_size=results['args']['hidden_size'], max_bin=state['input_mean'].shape[0])
            if pretrained:
                models[target].load_state_dict(state, strict=False)
            models[target].to(device)
        return models