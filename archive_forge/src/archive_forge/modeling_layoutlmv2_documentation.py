import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config

        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        In this example below, we give the LayoutLMv2 model an image (of texts) and ask it a question. It will give us
        a prediction of what it thinks the answer is (the span of the answer within the texts parsed from the image).

        ```python
        >>> from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering, set_seed
        >>> import torch
        >>> from PIL import Image
        >>> from datasets import load_dataset

        >>> set_seed(88)
        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")
        >>> question = "When is coffee break?"
        >>> encoding = processor(image, question, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> predicted_start_idx = outputs.start_logits.argmax(-1).item()
        >>> predicted_end_idx = outputs.end_logits.argmax(-1).item()
        >>> predicted_start_idx, predicted_end_idx
        (154, 287)

        >>> predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
        >>> predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
        >>> predicted_answer  # results are not very good without further fine-tuning
        'council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public ...
        ```

        ```python
        >>> target_start_index = torch.tensor([7])
        >>> target_end_index = torch.tensor([14])
        >>> outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)
        >>> predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
        >>> predicted_answer_span_end = outputs.end_logits.argmax(-1).item()
        >>> predicted_answer_span_start, predicted_answer_span_end
        (154, 287)
        ```
        