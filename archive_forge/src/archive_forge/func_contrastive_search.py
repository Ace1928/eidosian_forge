import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig
from .logits_process import (
from .stopping_criteria import (
@torch.no_grad()
def contrastive_search(self, input_ids: torch.LongTensor, top_k: Optional[int]=1, penalty_alpha: Optional[float]=0, logits_processor: Optional[LogitsProcessorList]=None, logits_warper: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, output_logits: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: bool=False, streamer: Optional['BaseStreamer']=None, sequential: Optional[bool]=None, **model_kwargs) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.contrastive_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            top_k (`int`, *optional*, defaults to 1):
                The size of the candidate set that is used to re-rank for contrastive search
            penalty_alpha (`float`, *optional*, defaults to 0):
                The degeneration penalty for contrastive search; activate when it is larger than 0
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            sequential (`bool`, *optional*):
                Switches topk hidden state computation from parallel to sequential to reduce memory if True.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:
        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "DeepMind Company is"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt")
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
        >>> outputs = model.contrastive_search(
        ...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\\n\\nIn this post, we talk about the benefits of deep learning in business and how it']
        ```"""
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    sequential = sequential if sequential is not None else self.generation_config.low_memory
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
    raw_logits = () if return_dict_in_generate and output_logits else None
    scores = () if return_dict_in_generate and output_scores else None
    decoder_attentions = () if return_dict_in_generate and output_attentions else None
    cross_attentions = () if return_dict_in_generate and output_attentions else None
    decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
        encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    this_peer_finished = False
    batch_size = input_ids.shape[0]
    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break
        if model_kwargs.get('past_key_values') is None:
            model_kwargs['use_cache'] = True
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
            if self.config.is_encoder_decoder:
                last_hidden_states = outputs.decoder_hidden_states[-1]
            else:
                last_hidden_states = outputs.hidden_states[-1]
            logit_for_next_step = outputs.logits[:, -1, :]
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder, standardize_cache_format=True)
            if not sequential:
                _, model_kwargs = self._expand_inputs_for_generation(expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            past_key_values = model_kwargs.get('past_key_values')
            if past_key_values is None:
                raise ValueError(f"{self.__class__.__name__} does not support caching and therefore **can't** be used for contrastive search.")
            elif not isinstance(past_key_values[0], (tuple, torch.Tensor)) or past_key_values[0][0].shape[0] != batch_size:
                raise ValueError(f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be used for contrastive search without further modifications.")
        processed_logit_for_next_step = logits_processor(input_ids, logit_for_next_step)
        processed_logit_for_next_step = logits_warper(input_ids, processed_logit_for_next_step)
        next_probs = nn.functional.softmax(processed_logit_for_next_step, dim=-1)
        top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)
        if return_dict_in_generate:
            if output_logits:
                raw_logits += (logit_for_next_step,)
            if output_scores:
                scores += (processed_logit_for_next_step,)
            if output_attentions:
                decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
        new_key_values = []
        for layer in model_kwargs['past_key_values']:
            items = []
            for item in layer:
                if sequential:
                    items.append(item.repeat_interleave(1, dim=0))
                else:
                    items.append(item.repeat_interleave(top_k, dim=0))
            new_key_values.append(tuple(items))
        model_kwargs['past_key_values'] = tuple(new_key_values)
        if sequential:
            all_outputs = []
            for i in range(top_k):
                next_model_inputs = self.prepare_inputs_for_generation(top_k_ids[:, i].view(-1, 1), **model_kwargs)
                outputs = self(**next_model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
                all_outputs.append(outputs)
            outputs = stack_model_outputs(all_outputs)
        else:
            next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)
            outputs = self(**next_model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions)
        if self.config.is_encoder_decoder:
            next_hidden = outputs.decoder_hidden_states[-1]
            full_hidden_states = outputs.decoder_hidden_states
        else:
            next_hidden = outputs.hidden_states[-1]
            full_hidden_states = outputs.hidden_states
        logits = outputs.logits[:, -1, :]
        context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)
        selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)
        selected_idx = selected_idx.to('cpu')
        next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
        next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
        next_hidden = next_hidden[range(batch_size), selected_idx, :]
        last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)
        next_decoder_hidden_states = ()
        for layer in full_hidden_states:
            layer = torch.stack(torch.split(layer, top_k))[range(batch_size), selected_idx, :]
            next_decoder_hidden_states += (layer,)
        if sequential:
            next_model_input = self.prepare_inputs_for_generation(top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs)
            selected_outputs = self(**next_model_input, return_dict=True, output_hidden_states=False, output_attentions=False)
            next_past_key_values = selected_outputs['past_key_values']
        else:
            next_past_key_values = self._extract_past_from_model_output(outputs, standardize_cache_format=True)
            new_key_values = ()
            for layer in next_past_key_values:
                items = ()
                for item in layer:
                    item = torch.stack(torch.split(item, top_k, dim=0))
                    item = item[range(batch_size), selected_idx, ...]
                    items += (item,)
                new_key_values += (items,)
            next_past_key_values = new_key_values
        logit_for_next_step = torch.stack(torch.split(logits, top_k))[range(batch_size), selected_idx, :]
        if self.config.is_encoder_decoder:
            next_step_cross_attentions = ()
            next_step_decoder_attentions = ()
            if output_attentions:
                for layer in outputs.cross_attentions:
                    layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                    next_step_cross_attentions += (layer,)
                for layer in outputs.decoder_attentions:
                    layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                    next_step_decoder_attentions += (layer,)
            outputs = Seq2SeqLMOutput(past_key_values=next_past_key_values, decoder_hidden_states=next_decoder_hidden_states, decoder_attentions=next_step_decoder_attentions or None, cross_attentions=next_step_cross_attentions or None)
        else:
            next_step_attentions = ()
            if output_attentions:
                for layer in outputs.attentions:
                    layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                    next_step_attentions += (layer,)
            outputs = CausalLMOutputWithPast(past_key_values=next_past_key_values, hidden_states=next_decoder_hidden_states, attentions=next_step_attentions or None)
        if synced_gpus and this_peer_finished:
            continue
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError('If `eos_token_id` is defined, make sure that `pad_token_id` is defined.')
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0))
            if unfinished_sequences.max() == 0:
                this_peer_finished = True
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True
        if this_peer_finished and (not synced_gpus):
            break
    if streamer is not None:
        streamer.end()
    if return_dict_in_generate:
        if model_kwargs.get('past_key_values') is not None:
            past_key_values = []
            for layer in model_kwargs['past_key_values']:
                layer_past_key_values = []
                for item in layer:
                    layer_past_key_values.append(item[..., :-1, :])
                past_key_values.append(tuple(layer_past_key_values))
            model_kwargs['past_key_values'] = tuple(past_key_values)
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(sequences=input_ids, scores=scores, logits=raw_logits, encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
        else:
            return GenerateDecoderOnlyOutput(sequences=input_ids, scores=scores, logits=raw_logits, attentions=decoder_attentions, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
    else:
        return input_ids