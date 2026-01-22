import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple
import torch
from audiocraft.models import MusicGen
from transformers import (
from transformers.models.musicgen_melody.configuration_musicgen_melody import MusicgenMelodyDecoderConfig
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
from transformers.models.musicgen_melody.processing_musicgen_melody import MusicgenMelodyProcessor
from transformers.utils import logging
@torch.no_grad()
def convert_musicgen_melody_checkpoint(checkpoint, pytorch_dump_folder=None, repo_id=None, device='cpu', test_same_output=False):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=args.device)
    decoder_config = decoder_config_from_checkpoint(checkpoint)
    decoder_state_dict = fairseq_model.lm.state_dict()
    decoder_state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict = rename_state_dict(decoder_state_dict, hidden_size=decoder_config.hidden_size)
    text_encoder = T5EncoderModel.from_pretrained('t5-base')
    audio_encoder = EncodecModel.from_pretrained('facebook/encodec_32khz')
    decoder = MusicgenMelodyForCausalLM(decoder_config).eval()
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)
    for key in missing_keys.copy():
        if key.startswith(('text_encoder', 'audio_encoder')) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)
    for key in unexpected_keys.copy():
        if key in EXPECTED_ADDITIONAL_KEYS:
            unexpected_keys.remove(key)
    if len(missing_keys) > 0:
        raise ValueError(f'Missing key(s) in state_dict: {missing_keys}')
    if len(unexpected_keys) > 0:
        raise ValueError(f'Unexpected key(s) in state_dict: {unexpected_keys}')
    model = MusicgenMelodyForConditionalGeneration(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder).to(args.device)
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)
    model.audio_enc_to_dec_proj.load_state_dict(audio_enc_to_dec_proj_state_dict)
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1).to(device)
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
    output_length = 1 + input_ids.shape[1] + model.config.chroma_length
    if logits.shape != (2 * decoder_config.num_codebooks, output_length, 2048):
        raise ValueError('Incorrect shape for logits')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    feature_extractor = MusicgenMelodyFeatureExtractor()
    processor = MusicgenMelodyProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0
    if test_same_output:
        decoder_input_ids = torch.ones_like(decoder_input_ids).to(device) * model.generation_config.pad_token_id
        with torch.no_grad():
            decoder_input_ids = decoder_input_ids[:decoder_config.num_codebooks]
            inputs = processor(text=['gen'], return_tensors='pt', padding=True).to(device)
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
            attributes, prompt_tokens = fairseq_model._prepare_tokens_and_attributes(['gen'], None)
            original_logits = fairseq_model.lm.forward(decoder_input_ids.reshape(1, decoder_config.num_codebooks, -1), attributes)
            torch.testing.assert_close(original_logits.squeeze(2).reshape(decoder_config.num_codebooks, -1), logits[:, -1], rtol=1e-05, atol=5e-05)
    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f'Saving model {checkpoint} to {pytorch_dump_folder}')
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)
    if repo_id:
        logger.info(f'Pushing model {checkpoint} to {repo_id}')
        model.push_to_hub(repo_id, create_pr=True)
        processor.push_to_hub(repo_id, create_pr=True)