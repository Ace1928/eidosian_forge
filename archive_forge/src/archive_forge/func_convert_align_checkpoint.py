import argparse
import os
import align
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image
from tokenizer import Tokenizer
from transformers import (
from transformers.utils import logging
@torch.no_grad()
def convert_align_checkpoint(checkpoint_path, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ALIGN structure.
    """
    seq_length = 64
    tok = Tokenizer(seq_length)
    original_model = align.Align('efficientnet-b7', 'bert-base', 640, seq_length, tok.get_vocab_size())
    original_model.compile()
    original_model.load_weights(checkpoint_path)
    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())
    config = get_align_config()
    hf_model = AlignModel(config).eval()
    hf_params = hf_model.state_dict()
    print('Converting parameters...')
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)
    processor = get_processor()
    inputs = processor(images=prepare_img(), text='A picture of a cat', padding='max_length', max_length=64, return_tensors='pt')
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)
    hf_image_features = outputs.image_embeds.detach().numpy()
    hf_text_features = outputs.text_embeds.detach().numpy()
    original_model.trainable = False
    tf_image_processor = EfficientNetImageProcessor(do_center_crop=True, do_rescale=False, do_normalize=False, include_top=False, resample=Image.BILINEAR)
    image = tf_image_processor(images=prepare_img(), return_tensors='tf', data_format='channels_last')['pixel_values']
    text = tok(tf.constant(['A picture of a cat']))
    image_features = original_model.image_encoder(image, training=False)
    text_features = original_model.text_encoder(text, training=False)
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    text_features = tf.nn.l2_normalize(text_features, axis=-1)
    if not np.allclose(image_features, hf_image_features, atol=0.001):
        raise ValueError('The predicted image features are not the same.')
    if not np.allclose(text_features, hf_text_features, atol=0.001):
        raise ValueError('The predicted text features are not the same.')
    print('Model outputs match!')
    if save_model:
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)
    if push_to_hub:
        print('Pushing converted ALIGN to the hub...')
        processor.push_to_hub('align-base')
        hf_model.push_to_hub('align-base')