import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:
            `MoECausalLMOutputWithPast` or `tuple` if `return_dict` returns MoECausalLMOutputWithPast insted of tuple

        Example:

        Text Generation with regular LM Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("織田信長は、", return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> gen_token = model.generate(input_ids, max_new_tokens=50)
        >>> tokenizer.decode(gen_token[0])
        "織田信長は、政治・軍事の中枢まで掌握した政治家であり、日本史上類を見ない驚異的な軍事侵攻を続け..."
        ```

        Text Generation with Prefix-LM Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("", prefix_text="織田信長は、", return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> token_type_ids = x_token.token_type_ids.to(device)
        >>> gen_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
        >>> tokenizer.decode(gen_token[0])
        "織田信長は、政治・外交で数々の戦果を上げるが、1568年からは、いわゆる本能寺の変で細川晴元に暗殺される..."
        ```

        Simultaneously Text Generation And Masked Language Model
        ```python
        >>> from transformers import AutoModel, AutoTokenizer, trainer_utils

        >>> device = "cuda"
        >>> model = AutoModel.from_pretrained("Tanrei/GPTSAN-japanese").to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> masked_sentence = "武田信玄は、<|inputmask|>時代ファンならぜひ押さえ<|inputmask|>きたい名将の一人。"
        >>> x_token = tokenizer("", prefix_text=masked_sentence, return_tensors="pt")
        >>> trainer_utils.set_seed(30)
        >>> input_ids = x_token.input_ids.to(device)
        >>> token_type_ids = x_token.token_type_ids.to(device)
        >>> out_lm_token = model.generate(input_ids, token_type_ids=token_type_ids, max_new_tokens=50)
        >>> out_mlm_token = model(input_ids, token_type_ids=token_type_ids).logits.argmax(axis=-1)
        >>> tokenizer.decode(out_mlm_token[0])
        "武田信玄は、戦国時代ファンならぜひ押さえておきたい名将の一人。"

        >>> tokenizer.decode(out_lm_token[0][input_ids.shape[1] :])
        "武田氏の三代に渡った武田家のひとり\n甲斐市に住む、日本史上最大の戦国大名。..."
        ```