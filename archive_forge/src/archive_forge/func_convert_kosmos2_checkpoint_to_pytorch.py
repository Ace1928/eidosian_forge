import argparse
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from transformers import Kosmos2Config, Kosmos2ForConditionalGeneration
def convert_kosmos2_checkpoint_to_pytorch(checkpoint_path, pytorch_dump_folder_path):
    state = load_checkpoint_to_cpu(checkpoint_path)
    state_dict = state['model']
    state_dict_keys = list(state_dict.keys())
    config = Kosmos2Config()
    config.text_config.no_repeat_ngram_size = 3
    model = Kosmos2ForConditionalGeneration(config)
    converted_state_dict = {}
    for key in state_dict_keys:
        if key in KEYS_TO_IGNORE:
            continue
        renamed_key = rename_key(key)
        converted_state_dict[renamed_key] = state_dict[key]
    model.load_state_dict(converted_state_dict, strict=True)
    model.save_pretrained(pytorch_dump_folder_path)