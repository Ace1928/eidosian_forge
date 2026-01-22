from functools import singledispatch
from outlines.fsm.guide import RegexGuide
from outlines.generate.api import SequenceGenerator
from outlines.integrations.llamacpp import RegexLogitsProcessor
from outlines.models import OpenAI
from outlines.models.llamacpp import LlamaCpp, LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial
@regex.register(LlamaCpp)
def regex_llamacpp(model: LlamaCpp, regex_str: str, sampler: Sampler=multinomial()):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError('The llama.cpp integration does not currently support any other sampling algorithm ' + 'than the multinomial sampler.')
    logits_processor = RegexLogitsProcessor(regex_str, llm=model.model)
    generator = LlamaSequenceGenerator(logits_processor=logits_processor, model=model)
    return generator