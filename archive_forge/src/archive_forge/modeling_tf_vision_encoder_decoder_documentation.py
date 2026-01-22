from __future__ import annotations
import re
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...modeling_tf_outputs import TFBaseModelOutput, TFSeq2SeqLMOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFPreTrainedModel, get_initializer, keras, unpack_inputs
from ...tf_utils import shape_list
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_tf_auto import TFAutoModel, TFAutoModelForCausalLM
from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoTokenizer, TFVisionEncoderDecoderModel
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> decoder_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> # initialize a bert2gpt2 from a pretrained BERT and GPT2 models. Note that the cross-attention layers will be randomly initialized
        >>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        ...     "google/vit-base-patch16-224-in21k", "openai-community/gpt2"
        ... )

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> img = Image.open(requests.get(url, stream=True).raw)

        >>> # forward
        >>> pixel_values = image_processor(images=img, return_tensors="tf").pixel_values  # Batch size 1
        >>> decoder_input_ids = decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids  # Batch size 1
        >>> outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

        >>> # training
        >>> outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        >>> loss, logits = outputs.loss, outputs.logits

        >>> # save and load from pretrained
        >>> model.save_pretrained("vit-gpt2")
        >>> model = TFVisionEncoderDecoderModel.from_pretrained("vit-gpt2")

        >>> # generation
        >>> generated = model.generate(pixel_values, decoder_start_token_id=model.config.decoder.bos_token_id)
        ```