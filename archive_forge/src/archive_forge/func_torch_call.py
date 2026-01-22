import random
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from accelerate import PartialState
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from ..import_utils import is_peft_available, is_unsloth_available, is_xpu_available
from ..trainer.model_config import ModelConfig
def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    batch = super().torch_call(examples)
    if self.instruction_template is None:
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch['labels'][i] == self.response_token_ids[0])[0]:
                if self.response_token_ids == batch['labels'][i][idx:idx + len(self.response_token_ids)].tolist():
                    response_token_ids_start_idx = idx
            if response_token_ids_start_idx is None:
                warnings.warn(f'Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.')
                batch['labels'][i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                batch['labels'][i, :response_token_ids_end_idx] = self.ignore_index
    else:
        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []
            for assistant_idx in np.where(batch['labels'][i] == self.response_token_ids[0])[0]:
                if self.response_token_ids == batch['labels'][i][assistant_idx:assistant_idx + len(self.response_token_ids)].tolist():
                    response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))
            if len(response_token_ids_idxs) == 0:
                warnings.warn(f'Could not find response key `{self.response_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.')
                batch['labels'][i, :] = self.ignore_index
            human_token_ids = self.instruction_token_ids
            for human_idx in np.where(batch['labels'][i] == human_token_ids[0])[0]:
                if human_token_ids == batch['labels'][i][human_idx:human_idx + len(human_token_ids)].tolist():
                    human_token_ids_idxs.append(human_idx)
            if len(human_token_ids_idxs) == 0:
                warnings.warn(f'Could not find instruction key `{self.instruction_template}` in the following instance: {self.tokenizer.decode(batch['input_ids'][i])} This instance will be ignored in loss calculation. Note, if this happens often, consider increasing the `max_seq_length`.')
                batch['labels'][i, :] = self.ignore_index
            if len(human_token_ids_idxs) > 0 and len(response_token_ids_idxs) > 0 and (human_token_ids_idxs[0] > response_token_ids_idxs[0]):
                human_token_ids_idxs = [0] + human_token_ids_idxs
            for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                if idx != 0:
                    batch['labels'][i, start:end] = self.ignore_index
                else:
                    batch['labels'][i, :end] = self.ignore_index
            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                batch['labels'][i, human_token_ids_idxs[-1]:] = self.ignore_index
    return batch