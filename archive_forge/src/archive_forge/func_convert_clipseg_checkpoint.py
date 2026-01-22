import argparse
import requests
import torch
from PIL import Image
from transformers import (
def convert_clipseg_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    config = get_clipseg_config(model_name)
    model = CLIPSegForImageSegmentation(config)
    model.eval()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    for key in state_dict.copy().keys():
        if key.startswith('model'):
            state_dict.pop(key, None)
    state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys != ['clip.text_model.embeddings.position_ids', 'clip.vision_model.embeddings.position_ids']:
        raise ValueError('Missing keys that are not expected: {}'.format(missing_keys))
    if unexpected_keys != ['decoder.reduce.weight', 'decoder.reduce.bias']:
        raise ValueError(f'Unexpected keys: {unexpected_keys}')
    image_processor = ViTImageProcessor(size=352)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    processor = CLIPSegProcessor(image_processor=image_processor, tokenizer=tokenizer)
    image = prepare_img()
    text = ['a glass', 'something to fill', 'wood', 'a jar']
    inputs = processor(text=text, images=[image] * len(text), padding='max_length', return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    expected_conditional = torch.tensor([0.111, -0.1882, 0.1645])
    expected_pooled_output = torch.tensor([0.2692, -0.7197, -0.1328])
    if model_name == 'clipseg-rd64-refined':
        expected_masks_slice = torch.tensor([[-10.0407, -9.9431, -10.2646], [-9.9751, -9.7064, -9.9586], [-9.6891, -9.5645, -9.9618]])
    elif model_name == 'clipseg-rd64':
        expected_masks_slice = torch.tensor([[-7.2877, -7.2711, -7.2463], [-7.2652, -7.278, -7.252], [-7.2239, -7.2204, -7.2001]])
    elif model_name == 'clipseg-rd16':
        expected_masks_slice = torch.tensor([[-6.3955, -6.4055, -6.4151], [-6.3911, -6.4033, -6.41], [-6.3474, -6.3702, -6.3762]])
    else:
        raise ValueError(f'Model name {model_name} not supported.')
    assert torch.allclose(outputs.logits[0, :3, :3], expected_masks_slice, atol=0.001)
    assert torch.allclose(outputs.conditional_embeddings[0, :3], expected_conditional, atol=0.001)
    assert torch.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=0.001)
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model and processor to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and processor for {model_name} to the hub')
        model.push_to_hub(f'CIDAS/{model_name}')
        processor.push_to_hub(f'CIDAS/{model_name}')