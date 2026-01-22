import logging
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra
@classmethod
def from_model_id_low_bit(cls, model_id: str, model_kwargs: Optional[dict]=None, **kwargs: Any) -> LLM:
    """
        Construct low_bit object from model_id

        Args:

            model_id: Path for the ipex-llm transformers low-bit model folder.
            model_kwargs: Keyword arguments to pass to the model and tokenizer.
            kwargs: Extra arguments to pass to the model and tokenizer.

        Returns:
            An object of IpexLLM.
        """
    try:
        from ipex_llm.transformers import AutoModel, AutoModelForCausalLM
        from transformers import AutoTokenizer, LlamaTokenizer
    except ImportError:
        raise ValueError('Could not import ipex-llm or transformers. Please install it with `pip install --pre --upgrade ipex-llm[all]`.')
    _model_kwargs = model_kwargs or {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
    except Exception:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)
    try:
        model = AutoModelForCausalLM.load_low_bit(model_id, **_model_kwargs)
    except Exception:
        model = AutoModel.load_low_bit(model_id, **_model_kwargs)
    if 'trust_remote_code' in _model_kwargs:
        _model_kwargs = {k: v for k, v in _model_kwargs.items() if k != 'trust_remote_code'}
    return cls(model_id=model_id, model=model, tokenizer=tokenizer, model_kwargs=_model_kwargs, **kwargs)