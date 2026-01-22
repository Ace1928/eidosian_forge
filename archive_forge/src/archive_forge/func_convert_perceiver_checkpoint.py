import argparse
import json
import pickle
from pathlib import Path
import haiku as hk
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture='MLM'):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """
    with open(pickle_file, 'rb') as f:
        checkpoint = pickle.loads(f.read())
    state = None
    if isinstance(checkpoint, dict) and architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        params = checkpoint['params']
        state = checkpoint['state']
    else:
        params = checkpoint
    state_dict = {}
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + '/' + param_name] = param
    if state is not None:
        for scope_name, parameters in hk.data_structures.to_mutable_dict(state).items():
            for param_name, param in parameters.items():
                state_dict[scope_name + '/' + param_name] = param
    rename_keys(state_dict, architecture=architecture)
    config = PerceiverConfig()
    subsampling = None
    repo_id = 'huggingface/label-files'
    if architecture == 'MLM':
        config.qk_channels = 8 * 32
        config.v_channels = 1280
        model = PerceiverForMaskedLM(config)
    elif 'image_classification' in architecture:
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        config.num_labels = 1000
        filename = 'imagenet-1k-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        if architecture == 'image_classification':
            config.image_size = 224
            model = PerceiverForImageClassificationLearned(config)
        elif architecture == 'image_classification_fourier':
            config.d_model = 261
            model = PerceiverForImageClassificationFourier(config)
        elif architecture == 'image_classification_conv':
            config.d_model = 322
            model = PerceiverForImageClassificationConvProcessing(config)
        else:
            raise ValueError(f'Architecture {architecture} not supported')
    elif architecture == 'optical_flow':
        config.num_latents = 2048
        config.d_latents = 512
        config.d_model = 322
        config.num_blocks = 1
        config.num_self_attends_per_block = 24
        config.num_self_attention_heads = 16
        config.num_cross_attention_heads = 1
        model = PerceiverForOpticalFlow(config)
    elif architecture == 'multimodal_autoencoding':
        config.num_latents = 28 * 28 * 1
        config.d_latents = 512
        config.d_model = 704
        config.num_blocks = 1
        config.num_self_attends_per_block = 8
        config.num_self_attention_heads = 8
        config.num_cross_attention_heads = 1
        config.num_labels = 700
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        nchunks = 128
        image_chunk_size = np.prod((16, 224, 224)) // nchunks
        audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
        chunk_idx = 0
        subsampling = {'image': torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)), 'audio': torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)), 'label': None}
        model = PerceiverForMultimodalAutoencoding(config)
        filename = 'kinetics700-id2label.json'
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        raise ValueError(f'Architecture {architecture} not supported')
    model.eval()
    model.load_state_dict(state_dict)
    input_mask = None
    if architecture == 'MLM':
        tokenizer = PerceiverTokenizer.from_pretrained('/Users/NielsRogge/Documents/Perceiver/Tokenizer files')
        text = 'This is an incomplete sentence where some words are missing.'
        encoding = tokenizer(text, padding='max_length', return_tensors='pt')
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs = encoding.input_ids
        input_mask = encoding.attention_mask
    elif architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        image_processor = PerceiverImageProcessor()
        image = prepare_img()
        encoding = image_processor(image, return_tensors='pt')
        inputs = encoding.pixel_values
    elif architecture == 'optical_flow':
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == 'multimodal_autoencoding':
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {'image': images, 'audio': audio, 'label': torch.zeros((images.shape[0], 700))}
    if architecture == 'multimodal_autoencoding':
        outputs = model(inputs=inputs, attention_mask=input_mask, subsampled_output_points=subsampling)
    else:
        outputs = model(inputs=inputs, attention_mask=input_mask)
    logits = outputs.logits
    if not isinstance(logits, dict):
        print('Shape of logits:', logits.shape)
    else:
        for k, v in logits.items():
            print(f'Shape of logits of modality {k}', v.shape)
    if architecture == 'MLM':
        expected_slice = torch.tensor([[-11.8336, -11.685, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.844, -12.641, -12.8646]])
        assert torch.allclose(logits[0, :3, :3], expected_slice)
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        expected_list = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        assert masked_tokens_predictions == expected_list
        print('Greedy predictions:')
        print(masked_tokens_predictions)
        print()
        print('Predicted string:')
        print(tokenizer.decode(masked_tokens_predictions))
    elif architecture in ['image_classification', 'image_classification_fourier', 'image_classification_conv']:
        print('Predicted class:', model.config.id2label[logits.argmax(-1).item()])
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)