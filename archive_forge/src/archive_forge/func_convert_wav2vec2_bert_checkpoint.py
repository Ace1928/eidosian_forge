import argparse
import torch
import torchaudio
from fairseq2.data import Collater
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.conformer_shaw import load_conformer_shaw_model
from transformers import (
@torch.no_grad()
def convert_wav2vec2_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, repo_id=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = Wav2Vec2BertConfig.from_pretrained(config_path, hidden_act='swish')
    else:
        config = Wav2Vec2BertConfig(apply_spec_augment=False)
    hf_wav2vec = Wav2Vec2BertModel(config)
    model = load_conformer_shaw_model(checkpoint_path, dtype=torch.float32)
    model.eval()
    hf_wav2vec = _convert_model(model, hf_wav2vec, wav2vec_convert_list)
    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        hf_wav2vec.push_to_hub(repo_id, create_pr=True)
    fe = SeamlessM4TFeatureExtractor(padding_value=1)
    fe._set_processor_class('Wav2Vec2BertProcessor')
    fe.save_pretrained(pytorch_dump_folder_path)
    if repo_id:
        fe.push_to_hub(repo_id, create_pr=True)
    if args.audio_path:
        waveform, sample_rate = torchaudio.load(args.audio_path)
        waveform = torchaudio.functional.resample(waveform, sample_rate, fe.sampling_rate)
        fbank_converter = WaveformToFbankConverter(num_mel_bins=80, waveform_scale=2 ** 15, channel_last=True, standardize=True, dtype=torch.float32)
        collater = Collater(pad_value=1)
        decoded_audio = {'waveform': waveform.T, 'sample_rate': fe.sampling_rate, 'format': -1}
        src = collater(fbank_converter(decoded_audio))['fbank']
        seqs, padding_mask = get_seqs_and_padding_mask(src)
        with torch.inference_mode():
            seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)
            original_output, padding_mask = model.encoder(seqs, padding_mask)
        hf_wav2vec.eval()
        inputs = fe(waveform, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = hf_wav2vec(**inputs)
        torch.testing.assert_close(original_output, outputs.last_hidden_state, atol=0.005, rtol=0.005)