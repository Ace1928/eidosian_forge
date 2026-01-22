import math
from typing import Optional
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ...bert.modeling_bert import BertModel
from .configuration_retribert import RetriBertConfig

        Args:
            input_ids_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the queries in a batch.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask_query (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            input_ids_doc (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the documents in a batch.
            attention_mask_doc (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on documents padding token indices.
            checkpoint_batch_size (`int`, *optional*, defaults to `-1`):
                If greater than 0, uses gradient checkpointing to only compute sequence representation on
                `checkpoint_batch_size` examples at a time on the GPU. All query representations are still compared to
                all document representations in the batch.

        Return:
            `torch.FloatTensor``: The bidirectional cross-entropy loss obtained while trying to match each query to its
            corresponding document and each document to its corresponding query in the batch
        