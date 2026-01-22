import re
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model
Builds :class:`Wav2Vec2Model` from the corresponding model object of
    `fairseq <https://github.com/pytorch/fairseq>`_.

    Args:
        original (torch.nn.Module):
            An instance of fairseq's Wav2Vec2.0 or HuBERT model.
            One of ``fairseq.models.wav2vec.wav2vec2_asr.Wav2VecEncoder``,
            ``fairseq.models.wav2vec.wav2vec2.Wav2Vec2Model`` or
            ``fairseq.models.hubert.hubert_asr.HubertEncoder``.

    Returns:
        Wav2Vec2Model: Imported model.

    Example - Loading pretrain-only model
        >>> from torchaudio.models.wav2vec2.utils import import_fairseq_model
        >>>
        >>> # Load model using fairseq
        >>> model_file = 'wav2vec_small.pt'
        >>> model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        >>> original = model[0]
        >>> imported = import_fairseq_model(original)
        >>>
        >>> # Perform feature extraction
        >>> waveform, _ = torchaudio.load('audio.wav')
        >>> features, _ = imported.extract_features(waveform)
        >>>
        >>> # Compare result with the original model from fairseq
        >>> reference = original.feature_extractor(waveform).transpose(1, 2)
        >>> torch.testing.assert_allclose(features, reference)

    Example - Fine-tuned model
        >>> from torchaudio.models.wav2vec2.utils import import_fairseq_model
        >>>
        >>> # Load model using fairseq
        >>> model_file = 'wav2vec_small_960h.pt'
        >>> model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
        >>> original = model[0]
        >>> imported = import_fairseq_model(original.w2v_encoder)
        >>>
        >>> # Perform encoding
        >>> waveform, _ = torchaudio.load('audio.wav')
        >>> emission, _ = imported(waveform)
        >>>
        >>> # Compare result with the original model from fairseq
        >>> mask = torch.zeros_like(waveform)
        >>> reference = original(waveform, mask)['encoder_out'].transpose(0, 1)
        >>> torch.testing.assert_allclose(emission, reference)
    