from functools import singledispatch
from outlines.fsm.guide import StopAtEOSGuide
from outlines.generate import SequenceGenerator
from outlines.models import LlamaCpp, OpenAI
from outlines.models.llamacpp import LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial
@text.register(OpenAI)
def text_openai(model: OpenAI, sampler: Sampler=multinomial()) -> OpenAI:
    if not isinstance(sampler, multinomial):
        raise NotImplementedError('The OpenAI API does not support any other sampling algorithm ' + 'than the multinomial sampler.')
    return model