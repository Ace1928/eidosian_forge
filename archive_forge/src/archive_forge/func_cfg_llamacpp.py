from functools import singledispatch
from outlines.fsm.guide import CFGGuide
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.models.llamacpp import (
from outlines.samplers import Sampler, multinomial
@cfg.register(LlamaCpp)
def cfg_llamacpp(model: LlamaCpp, cfg_str: str, sampler: Sampler=multinomial()):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError('The llama.cpp integration does not currently support any other sampling algorithm ' + 'than the multinomial sampler.')
    logits_processor = CFGLogitsProcessor(cfg_str, model.tokenizer)
    generator = LlamaSequenceGenerator(logits_processor, model)
    return generator