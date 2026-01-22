from __future__ import annotations
import re
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_utils import TFPreTrainedModel, keras, unpack_inputs
from ...tf_utils import shape_list
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel
from ..clip.modeling_tf_clip import CLIPVisionConfig, TFCLIPOutput, TFCLIPVisionModel
from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig

        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        