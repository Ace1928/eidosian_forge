from typing import Any, Dict, List, Mapping, Optional, Set
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.utils import enforce_stop_tokens
def rwkv_generate(self, prompt: str) -> str:
    self.model_state = None
    self.model_tokens = []
    logits = self.run_rnn(self.tokenizer.encode(prompt).ids)
    begin = len(self.model_tokens)
    out_last = begin
    occurrence: Dict = {}
    decoded = ''
    for i in range(self.max_tokens_per_generation):
        for n in occurrence:
            logits[n] -= self.penalty_alpha_presence + occurrence[n] * self.penalty_alpha_frequency
        token = self.pipeline.sample_logits(logits, temperature=self.temperature, top_p=self.top_p)
        END_OF_TEXT = 0
        if token == END_OF_TEXT:
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        logits = self.run_rnn([token])
        xxx = self.tokenizer.decode(self.model_tokens[out_last:])
        if '�' not in xxx:
            decoded += xxx
            out_last = begin + i + 1
            if i >= self.max_tokens_per_generation - 100:
                break
    return decoded