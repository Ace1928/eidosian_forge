import argparse
import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import (
@torch.no_grad()
def convert_grounding_dino_checkpoint(args):
    model_name = args.model_name
    pytorch_dump_folder_path = args.pytorch_dump_folder_path
    push_to_hub = args.push_to_hub
    verify_logits = args.verify_logits
    checkpoint_mapping = {'grounding-dino-tiny': 'https://huggingface.co/ShilongLiu/GroundingDino/resolve/main/groundingdino_swint_ogc.pth', 'grounding-dino-base': 'https://huggingface.co/ShilongLiu/GroundingDino/resolve/main/groundingdino_swinb_cogcoor.pth'}
    config = get_grounding_dino_config(model_name)
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location='cpu')['model']
    original_state_dict = {k.replace('module.', ''): v for k, v in original_state_dict.items()}
    for name, param in original_state_dict.items():
        print(name, param.shape)
    new_state_dict = original_state_dict.copy()
    rename_keys = create_rename_keys(original_state_dict, config)
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)
    read_in_q_k_v_encoder(new_state_dict, config)
    read_in_q_k_v_text_enhancer(new_state_dict, config)
    read_in_q_k_v_decoder(new_state_dict, config)
    model = GroundingDinoForObjectDetection(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
    image = prepare_img()
    transforms = T.Compose([T.Resize(size=800, max_size=1333), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    original_pixel_values = transforms(image).unsqueeze(0)
    image_processor = GroundingDinoImageProcessor()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    processor = GroundingDinoProcessor(image_processor=image_processor, tokenizer=tokenizer)
    text = 'a cat'
    inputs = processor(images=image, text=preprocess_caption(text), return_tensors='pt')
    assert torch.allclose(original_pixel_values, inputs.pixel_values, atol=0.0001)
    if verify_logits:
        with torch.no_grad():
            outputs = model(**inputs)
        print(outputs.logits[0, :3, :3])
        expected_slice = torch.tensor([[-4.8913, -0.19, -0.2161], [-4.9653, -0.3719, -0.395], [-5.9599, -3.3765, -3.3104]])
        assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=0.0001)
        print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        model.push_to_hub(f'EduardoPacheco/{model_name}')
        processor.push_to_hub(f'EduardoPacheco/{model_name}')