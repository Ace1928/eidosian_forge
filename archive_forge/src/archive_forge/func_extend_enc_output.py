import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
def extend_enc_output(tensor, num_beams=None):
    tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
    tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
    return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])