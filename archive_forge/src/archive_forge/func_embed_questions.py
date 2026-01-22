import math
from typing import Optional
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, logging
from ...bert.modeling_bert import BertModel
from .configuration_retribert import RetriBertConfig
def embed_questions(self, input_ids, attention_mask=None, checkpoint_batch_size=-1):
    q_reps = self.embed_sentences_checkpointed(input_ids, attention_mask, self.bert_query, checkpoint_batch_size)
    return self.project_query(q_reps)