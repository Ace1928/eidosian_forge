import argparse
import torch
from clip import load
from transformers import CLIPConfig, CLIPModel
@torch.no_grad()
def convert_clip_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    else:
        config = CLIPConfig(projection_dim=512, text_config={}, vision_config={})
    hf_model = CLIPModel(config).eval()
    pt_model, _ = load(checkpoint_path, device='cpu', jit=False)
    pt_model = pt_model.eval()
    copy_text_model_and_projection(hf_model, pt_model)
    copy_vison_model_and_projection(hf_model, pt_model)
    hf_model.logit_scale = pt_model.logit_scale
    input_ids = torch.arange(0, 77).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)
    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    hf_logits_per_image = hf_outputs.logits_per_image
    hf_logits_per_text = hf_outputs.logits_per_text
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)
    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=0.001)
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=0.001)
    hf_model.save_pretrained(pytorch_dump_folder_path)