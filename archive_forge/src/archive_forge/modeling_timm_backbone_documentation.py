from typing import Optional, Tuple, Union
import torch
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available, requires_backends
from ...utils.backbone_utils import BackboneMixin
from .configuration_timm_backbone import TimmBackboneConfig

        Empty init weights function to ensure compatibility of the class in the library.
        