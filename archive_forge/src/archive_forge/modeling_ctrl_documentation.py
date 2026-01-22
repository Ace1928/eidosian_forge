from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_ctrl import CTRLConfig

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example of single-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> model.config.id2label[predicted_class_id]
        'LABEL_0'
        ```

        ```python
        >>> import torch

        >>> torch.manual_seed(42)  # doctest: +IGNORE_RESULT
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

        >>> labels = torch.tensor(1)
        >>> loss = model(**inputs, labels=labels).loss
        >>> round(loss.item(), 2)
        0.35
        ```

        Example of multi-label classification:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CTRLForSequenceClassification

        >>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
        >>> model = CTRLForSequenceClassification.from_pretrained(
        ...     "Salesforce/ctrl", problem_type="multi_label_classification"
        ... )

        >>> # CTRL was trained with control codes as the first token
        >>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
        >>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> predicted_class_id = logits.argmax().item()
        >>> model.config.id2label[predicted_class_id]
        'LABEL_0'
        ```

        ```python
        >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
        >>> num_labels = len(model.config.id2label)
        >>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

        >>> num_labels = len(model.config.id2label)
        >>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
        ...     torch.float
        ... )
        >>> loss = model(**inputs, labels=labels).loss
        >>> loss.backward()  # doctest: +IGNORE_RESULT
        ```