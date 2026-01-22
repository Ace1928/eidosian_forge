import argparse
import json
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTImageProcessor, ViTMSNConfig, ViTMSNModel
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def convert_vit_msn_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    config = ViTMSNConfig()
    config.num_labels = 1000
    repo_id = 'datasets/huggingface/label-files'
    filename = 'imagenet-1k-id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename), 'r'))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if 's16' in checkpoint_url:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_attention_heads = 6
    elif 'l16' in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    elif 'b4' in checkpoint_url:
        config.patch_size = 4
    elif 'l7' in checkpoint_url:
        config.patch_size = 7
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.hidden_dropout_prob = 0.1
    model = ViTMSNModel(config)
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['target_encoder']
    image_processor = ViTImageProcessor(size=config.image_size)
    remove_projection_head(state_dict)
    rename_keys = create_rename_keys(config, base_model=True)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model=True)
    model.load_state_dict(state_dict)
    model.eval()
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTImageProcessor(size=config.image_size, image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD)
    inputs = image_processor(images=image, return_tensors='pt')
    torch.manual_seed(2)
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    if 's16' in checkpoint_url:
        expected_slice = torch.tensor([[-1.0915, -1.4876, -1.1809]])
    elif 'b16' in checkpoint_url:
        expected_slice = torch.tensor([[14.2889, -18.9045, 11.7281]])
    elif 'l16' in checkpoint_url:
        expected_slice = torch.tensor([[41.5028, -22.8681, 45.6475]])
    elif 'b4' in checkpoint_url:
        expected_slice = torch.tensor([[-4.3868, 5.2932, -0.4137]])
    else:
        expected_slice = torch.tensor([[-0.1792, -0.6465, 2.4263]])
    assert torch.allclose(last_hidden_state[:, 0, :3], expected_slice, atol=0.0001)
    print(f'Saving model to {pytorch_dump_folder_path}')
    model.save_pretrained(pytorch_dump_folder_path)
    print(f'Saving image processor to {pytorch_dump_folder_path}')
    image_processor.save_pretrained(pytorch_dump_folder_path)