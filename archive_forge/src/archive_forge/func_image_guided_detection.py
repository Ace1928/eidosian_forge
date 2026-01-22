import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
@add_start_docstrings_to_model_forward(OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2Config)
def image_guided_detection(self, pixel_values: torch.FloatTensor, query_pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Owlv2ImageGuidedObjectDetectionOutput:
    """
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy as np
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)
        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model.image_guided_detection(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = processor.post_process_image_guided_detection(
        ...     outputs=outputs, threshold=0.9, nms_threshold=0.3, target_sizes=target_sizes
        ... )
        >>> i = 0  # Retrieve predictions for the first image
        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]
        >>> for box, score in zip(boxes, scores):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
        Detected similar object with confidence 0.938 at location [490.96, 109.89, 821.09, 536.11]
        Detected similar object with confidence 0.959 at location [8.67, 721.29, 928.68, 732.78]
        Detected similar object with confidence 0.902 at location [4.27, 720.02, 941.45, 761.59]
        Detected similar object with confidence 0.985 at location [265.46, -58.9, 1009.04, 365.66]
        Detected similar object with confidence 1.0 at location [9.79, 28.69, 937.31, 941.64]
        Detected similar object with confidence 0.998 at location [869.97, 58.28, 923.23, 978.1]
        Detected similar object with confidence 0.985 at location [309.23, 21.07, 371.61, 932.02]
        Detected similar object with confidence 0.947 at location [27.93, 859.45, 969.75, 915.44]
        Detected similar object with confidence 0.996 at location [785.82, 41.38, 880.26, 966.37]
        Detected similar object with confidence 0.998 at location [5.08, 721.17, 925.93, 998.41]
        Detected similar object with confidence 0.969 at location [6.7, 898.1, 921.75, 949.51]
        Detected similar object with confidence 0.966 at location [47.16, 927.29, 981.99, 942.14]
        Detected similar object with confidence 0.924 at location [46.4, 936.13, 953.02, 950.78]
        ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.return_dict
    query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
    feature_map, vision_outputs = self.image_embedder(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
    batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
    image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
    batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
    query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
    query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats, query_feature_map)
    pred_logits, class_embeds = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)
    target_pred_boxes = self.box_predictor(image_feats, feature_map)
    if not return_dict:
        output = (feature_map, query_feature_map, target_pred_boxes, query_pred_boxes, pred_logits, class_embeds, vision_outputs.to_tuple())
        output = tuple((x for x in output if x is not None))
        return output
    return Owlv2ImageGuidedObjectDetectionOutput(image_embeds=feature_map, query_image_embeds=query_feature_map, target_pred_boxes=target_pred_boxes, query_pred_boxes=query_pred_boxes, logits=pred_logits, class_embeds=class_embeds, text_model_output=None, vision_model_output=vision_outputs)