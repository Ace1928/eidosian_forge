import argparse
import torch
from transformers import (
@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    downstream_dict = checkpoint['Downstream']
    hf_config = WavLMConfig.from_pretrained(config_path)
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name, return_attention_mask=True, do_normalize=False)
    arch = hf_config.architectures[0]
    if arch.endswith('ForSequenceClassification'):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith('ForAudioFrameClassification'):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith('ForXVector'):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        raise NotImplementedError(f'S3PRL weights conversion is not supported for {arch}')
    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint['Featurizer']['weights']
    hf_feature_extractor.save_pretrained(model_dump_path)
    hf_model.save_pretrained(model_dump_path)