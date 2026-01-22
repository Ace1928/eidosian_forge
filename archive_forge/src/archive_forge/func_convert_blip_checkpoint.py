import argparse
import re
import requests
import torch
from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import (
@torch.no_grad()
def convert_blip_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = BlipConfig.from_pretrained(config_path)
    else:
        config = BlipConfig(projection_dim=512, text_config={}, vision_config={})
    hf_model = BlipForConditionalGeneration(config).eval()
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    pt_model = blip_decoder(pretrained=model_url, image_size=384, vit='base')
    pt_model = pt_model.eval()
    modified_state_dict = pt_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value
    hf_model.load_state_dict(modified_state_dict)
    image_size = 384
    image = load_demo_image(image_size=image_size, device='cpu')
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    input_ids = tokenizer(['a picture of']).input_ids
    out = hf_model.generate(image, input_ids)
    assert out[0].tolist() == [30522, 1037, 3861, 1997, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]
    out = hf_model.generate(image)
    assert out[0].tolist() == [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]
    if pytorch_dump_folder_path is not None:
        hf_model.save_pretrained(pytorch_dump_folder_path)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    vqa_model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    vqa_model.eval()
    modified_state_dict = vqa_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value
    hf_vqa_model = BlipForQuestionAnswering(config)
    hf_vqa_model.load_state_dict(modified_state_dict)
    question = ['How many dogs are in this image?']
    question_input_ids = tokenizer(question, return_tensors='pt').input_ids
    answer = hf_vqa_model.generate(question_input_ids, image)
    print(tokenizer.decode(answer[0]))
    assert tokenizer.decode(answer[0]) == '[UNK] 1 [SEP]'
    if pytorch_dump_folder_path is not None:
        hf_vqa_model.save_pretrained(pytorch_dump_folder_path + '_vqa')
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    itm_model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    itm_model.eval()
    modified_state_dict = itm_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value
    hf_itm_model = BlipForImageTextRetrieval(config)
    question = ['A picture of a woman with a dog sitting in a beach']
    question_input_ids = tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=35).input_ids
    hf_itm_model.load_state_dict(modified_state_dict)
    hf_itm_model.eval()
    out_itm = hf_itm_model(question_input_ids, image, use_itm_head=True)
    out = hf_itm_model(question_input_ids, image, use_itm_head=False)
    assert out[0].item() == 0.2110687494277954
    assert torch.nn.functional.softmax(out_itm[0], dim=1)[:, 1].item() == 0.45698845386505127
    if pytorch_dump_folder_path is not None:
        hf_itm_model.save_pretrained(pytorch_dump_folder_path + '_itm')