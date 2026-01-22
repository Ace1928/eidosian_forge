import argparse
import json
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def convert_videomae_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    config = get_videomae_config(model_name)
    if 'finetuned' in model_name:
        model = VideoMAEForVideoClassification(config)
    else:
        model = VideoMAEForPreTraining(config)
    output = 'pytorch_model.bin'
    gdown.cached_download(checkpoint_url, output, quiet=False)
    files = torch.load(output, map_location='cpu')
    if 'model' in files:
        state_dict = files['model']
    else:
        state_dict = files['module']
    new_state_dict = convert_state_dict(state_dict, config)
    model.load_state_dict(new_state_dict)
    model.eval()
    image_processor = VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    video = prepare_video()
    inputs = image_processor(video, return_tensors='pt')
    if 'finetuned' not in model_name:
        local_path = hf_hub_download(repo_id='hf-internal-testing/bool-masked-pos', filename='bool_masked_pos.pt')
        inputs['bool_masked_pos'] = torch.load(local_path)
    outputs = model(**inputs)
    logits = outputs.logits
    model_names = ['videomae-small-finetuned-kinetics', 'videomae-small-finetuned-ssv2', 'videomae-base-short', 'videomae-base-short-finetuned-kinetics', 'videomae-base', 'videomae-base-finetuned-kinetics', 'videomae-large', 'videomae-large-finetuned-kinetics', 'videomae-huge-finetuned-kinetics', 'videomae-base-short-ssv2', 'videomae-base-short-finetuned-ssv2', 'videomae-base-ssv2', 'videomae-base-finetuned-ssv2']
    if model_name == 'videomae-small-finetuned-kinetics':
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9291, -0.4061, -0.9307])
    elif model_name == 'videomae-small-finetuned-ssv2':
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.2671, -0.4689, -0.8235])
    elif model_name == 'videomae-base':
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7739, 0.7968, 0.7089], [0.6701, 0.7487, 0.6209], [0.4287, 0.5158, 0.4773]])
    elif model_name == 'videomae-base-short':
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7994, 0.9612, 0.8508], [0.7401, 0.8958, 0.8302], [0.5862, 0.7468, 0.7325]])
        expected_loss = torch.tensor([0.5142]) if config.norm_pix_loss else torch.tensor([0.6469])
    elif model_name == 'videomae-large':
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7149, 0.7997, 0.6966], [0.6768, 0.7869, 0.6948], [0.5139, 0.6221, 0.5605]])
    elif model_name == 'videomae-large-finetuned-kinetics':
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.0771, 0.0011, -0.3625])
    elif model_name == 'videomae-huge-finetuned-kinetics':
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.2433, 0.1632, -0.4894])
    elif model_name == 'videomae-base-short-finetuned-kinetics':
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.6588, 0.099, -0.2493])
    elif model_name == 'videomae-base-finetuned-kinetics':
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0.3669, -0.0688, -0.2421])
    elif model_name == 'videomae-base-short-ssv2':
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.4712, 0.5296, 0.5786], [0.2278, 0.2729, 0.4026], [0.0352, 0.073, 0.2506]])
    elif model_name == 'videomae-base-short-finetuned-ssv2':
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.0537, -0.1539, -0.3266])
    elif model_name == 'videomae-base-ssv2':
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.8131, 0.8727, 0.8546], [0.7366, 0.9377, 0.887], [0.5935, 0.8874, 0.8564]])
    elif model_name == 'videomae-base-finetuned-ssv2':
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.1961, -0.8337, -0.6389])
    else:
        raise ValueError(f'Model name not supported. Should be one of {model_names}')
    assert logits.shape == expected_shape
    if 'finetuned' in model_name:
        assert torch.allclose(logits[0, :3], expected_slice, atol=0.0001)
    else:
        print('Logits:', logits[0, :3, :3])
        assert torch.allclose(logits[0, :3, :3], expected_slice, atol=0.0001)
    print('Logits ok!')
    if model_name == 'videomae-base-short':
        loss = outputs.loss
        assert torch.allclose(loss, expected_loss, atol=0.0001)
        print('Loss ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and image processor to {pytorch_dump_folder_path}')
        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing to the hub...')
        model.push_to_hub(model_name, organization='nielsr')