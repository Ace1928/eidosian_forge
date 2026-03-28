import torch
import pytest
from neural_forge.core.modality import ModalityClassifier, Modality, SharedLatentSpace
from neural_forge.core.tokenizer import ByteLevelTokenizer

def test_byte_tokenizer():
    """Verify byte-level encoding and decoding."""
    tokenizer = ByteLevelTokenizer()
    text = "Eidosian Forge 💎"
    tokens = tokenizer.encode(text)
    assert tokens.shape[1] > 0
    decoded = tokenizer.decode(tokens)
    assert decoded == text

def test_modality_routing():
    """Ensure modality classifier returns valid logits for all classes."""
    classifier = ModalityClassifier(768)
    x = torch.randn(1, 10, 768)
    logits = classifier(x)
    assert logits.shape == (1, 3)
