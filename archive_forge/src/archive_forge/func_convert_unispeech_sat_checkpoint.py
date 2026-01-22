import argparse
import fairseq
import torch
from transformers import UniSpeechSatConfig, UniSpeechSatForCTC, UniSpeechSatForPreTraining, logging
@torch.no_grad()
def convert_unispeech_sat_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = UniSpeechSatConfig.from_pretrained(config_path)
    else:
        config = UniSpeechSatConfig()
    dict_path = ''
    if is_finetuned:
        hf_wav2vec = UniSpeechSatForCTC(config)
    else:
        hf_wav2vec = UniSpeechSatForPreTraining(config)
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], arg_overrides={'data': '/'.join(dict_path.split('/')[:-1])})
    model = model[0].eval()
    recursively_load_weights(model, hf_wav2vec)
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)