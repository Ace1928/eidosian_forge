from typing import List, Optional, Tuple, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import make_async, LRUCache
from vllm.transformers_utils.tokenizers import *
def detokenize_incrementally(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], all_input_ids: List[int], prev_tokens: Optional[List[str]], prefix_offset: int=0, read_offset: int=0, skip_special_tokens: bool=False, spaces_between_special_tokens: bool=True) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]
    if prev_tokens is None:
        new_tokens = tokenizer.convert_ids_to_tokens(all_input_ids, skip_special_tokens=skip_special_tokens)
        output_tokens = new_tokens
        prefix_offset = max(len(output_tokens) - 6, 0)
        if skip_special_tokens and new_token_id in tokenizer.all_special_ids:
            read_offset = max(len(output_tokens), 0)
        else:
            read_offset = max(len(output_tokens) - 1, 0)
    else:
        new_tokens = tokenizer.convert_ids_to_tokens([new_token_id], skip_special_tokens=skip_special_tokens)
        output_tokens = prev_tokens + new_tokens
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(tokenizer, output_tokens[prefix_offset:read_offset], skip_special_tokens=skip_special_tokens, spaces_between_special_tokens=spaces_between_special_tokens)
        new_text = _convert_tokens_to_string_with_added_encoders(tokenizer, output_tokens[prefix_offset:], skip_special_tokens=skip_special_tokens, spaces_between_special_tokens=spaces_between_special_tokens)
    if len(new_text) > len(prefix_text) and (not new_text.endswith('�')):
        new_text = new_text[len(prefix_text):]
        return (new_tokens, new_text, read_offset, len(output_tokens))
    else:
        return (new_tokens, '', prefix_offset, read_offset)