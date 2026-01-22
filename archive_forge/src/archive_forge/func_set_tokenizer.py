from typing import List, Optional, Union
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
def set_tokenizer(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None:
    self.llm_engine.tokenizer.tokenizer = tokenizer