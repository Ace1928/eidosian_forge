import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def load_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    """
    from transformers import AutoBackbone, AutoConfig
    backbone_config = getattr(config, 'backbone_config', None)
    use_timm_backbone = getattr(config, 'use_timm_backbone', None)
    use_pretrained_backbone = getattr(config, 'use_pretrained_backbone', None)
    backbone_checkpoint = getattr(config, 'backbone', None)
    backbone_kwargs = getattr(config, 'backbone_kwargs', None)
    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")
    if backbone_config is not None and backbone_checkpoint is not None and (use_pretrained_backbone is not None):
        raise ValueError('Cannot specify both config.backbone_config and config.backbone')
    if backbone_config is None and use_timm_backbone is None and (backbone_checkpoint is None) and (backbone_checkpoint is None):
        return AutoBackbone.from_config(config=config, **backbone_kwargs)
    if use_timm_backbone:
        if backbone_checkpoint is None:
            raise ValueError('config.backbone must be set if use_timm_backbone is True')
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, use_timm_backbone=use_timm_backbone, use_pretrained_backbone=use_pretrained_backbone, **backbone_kwargs)
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            raise ValueError('config.backbone must be set if use_pretrained_backbone is True')
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            raise ValueError('Either config.backbone_config or config.backbone must be set')
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        backbone = AutoBackbone.from_config(config=backbone_config)
    return backbone