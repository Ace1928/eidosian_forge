import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.tokenizer import TokenizerGroup, get_lora_tokenizer
def test_get_lora_tokenizer(sql_lora_files, tmpdir):
    lora_request = None
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer
    lora_request = LoRARequest('1', 1, sql_lora_files)
    tokenizer = get_lora_tokenizer(lora_request)
    assert tokenizer.get_added_vocab()
    lora_request = LoRARequest('1', 1, str(tmpdir))
    tokenizer = get_lora_tokenizer(lora_request)
    assert not tokenizer