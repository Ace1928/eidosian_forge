import argparse
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def convert_vipllava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken('<image>', special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    config = VipLlavaConfig(text_config=text_config)
    config.pad_token_id = 32001
    with torch.device('meta'):
        model = VipLlavaForConditionalGeneration(config)
    pad_shape = 64
    state_dict_path = hf_hub_download(old_state_dict_id, 'model_state_dict_7b.bin')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = (pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-05 * sigma)
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))), dim=0)
    model.language_model.lm_head.weight.data[32000:] = torch.stack(tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))), dim=0)
    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)