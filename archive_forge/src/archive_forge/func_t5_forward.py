from typing import Optional, Tuple
import torch
def t5_forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False, output_attentions=False, **kwargs):
    raise_on_head_mask(layer_head_mask)
    if output_attentions is True:
        raise ValueError('output_attentions=True can not be supported with BetterTransformer.')
    if len(self.pruned_heads) > 0:
        raise ValueError(f'Setting `pruned_heads` is unsupported with BetterTransformer, found {self.pruned_heads}.')
    batch_size, seq_length = hidden_states.shape[:2]
    real_seq_length = seq_length
    if past_key_value is not None:
        assert len(past_key_value) == 2, f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states'
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length
    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            hidden_states = shape(proj_layer(key_value_states))
        if past_key_value is not None:
            if key_value_states is None:
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            elif past_key_value.shape[2] != key_value_states.shape[1]:
                hidden_states = shape(proj_layer(key_value_states))
            else:
                hidden_states = past_key_value
        return hidden_states
    query_states = shape(self.q(hidden_states))
    key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
    value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
    dropout_p = self.dropout if self.training else 0.0
    query_states = self.scale * query_states
    if position_bias is None and (not self.has_relative_attention_bias):
        if mask is None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=dropout_p, is_causal=False)
        elif mask is not None:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask, dropout_p=dropout_p, is_causal=False)
    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros((1, self.n_heads, real_seq_length, key_length), device=value_states.device, dtype=value_states.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=value_states.device)
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]
        if mask is not None:
            position_bias = position_bias + mask
        if self.has_relative_attention_bias:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False)
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=position_bias, dropout_p=dropout_p, is_causal=False)
    attn_output = unshape(attn_output)
    attn_output = self.o(attn_output)
    present_key_value_state = (key_states, value_states) if self.is_decoder and use_cache else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
    return outputs