import argparse
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from transformers import (
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    qformer_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', truncation_side='left')
    qformer_tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    if 't5' in model_name:
        tokenizer = T5TokenizerFast.from_pretrained('google/flan-t5-xl', truncation_side='left')
    elif 'vicuna' in model_name:
        tokenizer = LlamaTokenizerFast.from_pretrained('huggyllama/llama-7b', truncation_side='left', bos_token='</s>', unk_token='</s>')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config, image_size = get_blip2_config(model_name)
    hf_model = InstructBlipForConditionalGeneration(config).eval()
    model_name_to_original = {'instructblip-vicuna-7b': ('blip2_vicuna_instruct', 'vicuna7b'), 'instructblip-vicuna-13b': ('blip2_vicuna_instruct', 'vicuna13b'), 'instructblip-flan-t5-xl': ('blip2_t5_instruct', 'flant5xl'), 'instructblip-flan-t5-xxl': ('blip2_t5_instruct', 'flant5xxl')}
    name, type = model_name_to_original[model_name]
    print('Loading original model...')
    hf_model_device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    lavis_device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    original_model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=type, is_eval=True, device=lavis_device)
    original_model.eval()
    print('Done!')
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith('Qformer.bert'):
            key = key.replace('Qformer.bert', 'qformer')
        if 'attention.self' in key:
            key = key.replace('self', 'attention')
        if 'llm_proj' in key:
            key = key.replace('llm_proj', 'language_projection')
        if 't5_proj' in key:
            key = key.replace('t5_proj', 'language_projection')
        if key.startswith('llm_model'):
            key = key.replace('llm_model', 'language_model')
        if key.startswith('t5'):
            key = key.replace('t5', 'language')
        state_dict[key] = val
    read_in_q_v_bias(state_dict, config)
    hf_model.load_state_dict(state_dict, strict=True)
    image = load_demo_image()
    prompt = 'What is unusual about this image?'
    image_processor = BlipImageProcessor(size={'height': image_size, 'width': image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD)
    processor = InstructBlipProcessor(image_processor=image_processor, tokenizer=tokenizer, qformer_tokenizer=qformer_tokenizer)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(hf_model_device)
    original_pixel_values = vis_processors['eval'](image).unsqueeze(0).to(lavis_device)
    pixel_values = inputs.pixel_values
    assert torch.allclose(original_pixel_values.to(pixel_values.device), pixel_values)
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    with torch.no_grad():
        if 'vicuna' in model_name:
            original_logits = original_model({'image': original_pixel_values, 'text_input': [prompt]}).logits
            logits = hf_model(**inputs).logits
        else:
            original_logits = original_model({'image': original_pixel_values, 'text_input': [prompt], 'text_output': ['\n']}).logits
            label_input_ids = tokenizer('\n', return_tensors='pt').input_ids.to(hf_model_device)
            labels = label_input_ids.masked_fill(label_input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(**inputs, labels=labels).logits
    print('First values of original logits:', original_logits[0, :3, :3])
    print('First values of HF logits:', logits[0, :3, :3])
    assert original_logits.shape == logits.shape
    atol = 0.0001 if 'vicuna' in model_name else 1e-05
    assert torch.allclose(original_logits.to(logits.device), logits, atol=atol)
    print('Looks ok!')
    print('Generating with original model...')
    original_outputs = original_model.generate({'image': original_pixel_values, 'prompt': prompt}, num_beams=5)
    print('Generating with HF model...')
    outputs = hf_model.generate(**inputs, do_sample=False, num_beams=5, max_length=256, min_length=1, top_p=0.9, repetition_penalty=1.5, length_penalty=1.0, temperature=1)
    if 'vicuna' in model_name:
        outputs[outputs == 0] = 2
    print('Original generation:', original_outputs)
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]
    print('HF generation:', output_text)
    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        processor.push_to_hub(f'Salesforce/{model_name}')
        hf_model.push_to_hub(f'Salesforce/{model_name}')