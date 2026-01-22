import argparse
import requests
import torch
from PIL import Image
from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor
def convert_swin_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    config = get_swin_config(model_name)
    model = SwinForMaskedImageModeling(config)
    model.eval()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image_processor = ViTImageProcessor(size={'height': 192, 'width': 192})
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs).logits
    print(outputs.keys())
    print('Looks ok!')
    if pytorch_dump_folder_path is not None:
        print(f'Saving model {model_name} to {pytorch_dump_folder_path}')
        model.save_pretrained(pytorch_dump_folder_path)
        print(f'Saving image processor to {pytorch_dump_folder_path}')
        image_processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print(f'Pushing model and image processor for {model_name} to hub')
        model.push_to_hub(f'microsoft/{model_name}')
        image_processor.push_to_hub(f'microsoft/{model_name}')