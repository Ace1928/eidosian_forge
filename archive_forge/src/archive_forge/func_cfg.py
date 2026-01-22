from functools import singledispatch
from outlines.fsm.guide import CFGGuide
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.models.llamacpp import (
from outlines.samplers import Sampler, multinomial
@singledispatch
def cfg(model, cfg_str: str, sampler: Sampler=multinomial()) -> SequenceGenerator:
    """Generate text in the language of a Context-Free Grammar

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text.

    """
    fsm = CFGGuide(cfg_str, model.tokenizer)
    device = model.device
    generator = SequenceGenerator(fsm, model, sampler, device)
    return generator