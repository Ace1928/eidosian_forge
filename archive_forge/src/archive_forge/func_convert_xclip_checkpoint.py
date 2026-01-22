import argparse
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
def convert_xclip_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    model_to_url = {'xclip-base-patch32': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_8.pth', 'xclip-base-patch32-16-frames': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_16.pth', 'xclip-base-patch16': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_16_8.pth', 'xclip-base-patch16-16-frames': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_16_16.pth', 'xclip-large-patch14': 'https://drive.google.com/u/0/uc?id=1NUOImq0o5DlQTST17iIP3vG7DgmHQuCx&amp;export=download&amp;confirm=t&amp;uuid=b26caedc-88e2-473e-830a-9d158b653cdb', 'xclip-large-patch14-16-frames': 'https://drive.google.com/u/0/uc?id=1FOYgnJc097OJ4lGwtRCCydQyVPJEOH7d&amp;export=download&amp;confirm=t&amp;uuid=538fa810-e671-4050-b385-9a623f89804f', 'xclip-base-patch16-kinetics-600': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k600_16_8.pth', 'xclip-base-patch16-kinetics-600-16-frames': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k600_16_16.pth', 'xclip-large-patch14-kinetics-600': 'https://drive.google.com/u/0/uc?id=1FV8C1INuM91sLAN4ImjzePLIlpMSihwV&amp;export=download&amp;confirm=t&amp;uuid=141d4977-4a65-44ae-864f-4b0c19f838be', 'xclip-base-patch16-hmdb-2-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_2.pth', 'xclip-base-patch16-hmdb-4-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_4.pth', 'xclip-base-patch16-hmdb-8-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_8.pth', 'xclip-base-patch16-hmdb-16-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_16.pth', 'xclip-base-patch16-ucf-2-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_2.pth', 'xclip-base-patch16-ucf-4-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_4.pth', 'xclip-base-patch16-ucf-8-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_8.pth', 'xclip-base-patch16-ucf-16-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_16.pth', 'xclip-base-patch16-zero-shot': 'https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/zero.pth'}
    checkpoint_url = model_to_url[model_name]
    num_frames = 8
    if '16-frames' in model_name:
        num_frames = 16
    elif 'shot' in model_name:
        num_frames = 32
    config = get_xclip_config(model_name, num_frames)
    model = XCLIPModel(config)
    model.eval()
    if 'drive' in checkpoint_url:
        output = 'pytorch_model.bin'
        gdown.cached_download(checkpoint_url, output, quiet=False)
        state_dict = torch.load(output, map_location='cpu')['model']
    else:
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)['model']
    state_dict = convert_state_dict(state_dict, config)
    model = XCLIPModel(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['text_model.embeddings.position_ids', 'vision_model.embeddings.position_ids']
    model.eval()
    size = 336 if model_name == 'xclip-large-patch14-16-frames' else 224
    image_processor = VideoMAEImageProcessor(size=size)
    slow_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    fast_tokenizer = CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch32')
    processor = XCLIPProcessor(image_processor=image_processor, tokenizer=fast_tokenizer)
    video = prepare_video(num_frames)
    inputs = processor(text=['playing sports', 'eating spaghetti', 'go shopping'], videos=video, return_tensors='pt', padding=True)
    print('Shape of pixel values:', inputs.pixel_values.shape)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_video = outputs.logits_per_video
    probs = logits_per_video.softmax(dim=1)
    print('Probs:', probs)
    if model_name == 'xclip-base-patch32':
        expected_probs = torch.tensor([[0.0019, 0.9951, 0.003]])
    elif model_name == 'xclip-base-patch32-16-frames':
        expected_probs = torch.tensor([[0.00070999, 0.99883, 0.0004558]])
    elif model_name == 'xclip-base-patch16':
        expected_probs = torch.tensor([[0.0083, 0.9681, 0.0236]])
    elif model_name == 'xclip-base-patch16-16-frames':
        expected_probs = torch.tensor([[0.00076937, 0.99728, 0.0019473]])
    elif model_name == 'xclip-large-patch14':
        expected_probs = torch.tensor([[0.0062, 0.9864, 0.0075]])
    elif model_name == 'xclip-large-patch14-16-frames':
        expected_probs = torch.tensor([[0.00033877, 0.99937, 0.00028888]])
    elif model_name == 'xclip-base-patch16-kinetics-600':
        expected_probs = torch.tensor([[0.0555, 0.8914, 0.0531]])
    elif model_name == 'xclip-base-patch16-kinetics-600-16-frames':
        expected_probs = torch.tensor([[0.00038554, 0.99929, 0.00032754]])
    elif model_name == 'xclip-large-patch14-kinetics-600':
        expected_probs = torch.tensor([[0.0036, 0.992, 0.0045]])
    elif model_name == 'xclip-base-patch16-hmdb-2-shot':
        expected_probs = torch.tensor([[7.189e-06, 0.99994, 5.6559e-05]])
    elif model_name == 'xclip-base-patch16-hmdb-4-shot':
        expected_probs = torch.tensor([[1.032e-05, 0.99993, 6.2435e-05]])
    elif model_name == 'xclip-base-patch16-hmdb-8-shot':
        expected_probs = torch.tensor([[4.1377e-06, 0.9999, 9.8386e-05]])
    elif model_name == 'xclip-base-patch16-hmdb-16-shot':
        expected_probs = torch.tensor([[4.1347e-05, 0.99962, 0.00033411]])
    elif model_name == 'xclip-base-patch16-ucf-2-shot':
        expected_probs = torch.tensor([[8.5857e-05, 0.99928, 0.00063291]])
    elif model_name == 'xclip-base-patch16-ucf-4-shot':
        expected_probs = torch.tensor([[8.5857e-05, 0.99928, 0.00063291]])
    elif model_name == 'xclip-base-patch16-ucf-8-shot':
        expected_probs = torch.tensor([[0.0027, 0.9904, 0.007]])
    elif model_name == 'xclip-base-patch16-ucf-16-shot':
        expected_probs = torch.tensor([[0.00098219, 0.99593, 0.0030863]])
    elif model_name == 'xclip-base-patch16-zero-shot':
        expected_probs = torch.tensor([[0.00035082, 0.99785, 0.0017966]])
    else:
        raise ValueError(f'Model name {model_name} not supported')
    assert torch.allclose(probs, expected_probs, atol=0.001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing model, processor and slow tokenizer files to the hub...')
        model.push_to_hub(model_name, organization='nielsr')
        processor.push_to_hub(model_name, organization='nielsr')
        slow_tokenizer.push_to_hub(model_name, organization='nielsr')