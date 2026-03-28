from __future__ import annotations

import torch

from neural_forge.core.modality import Modality
from neural_forge.models.integrated import EidosianSingularityModel
from neural_forge.models.singularity import NeuralForgeSingularity


def test_singularity_forward_embedded_accepts_projected_features() -> None:
    model = NeuralForgeSingularity(vocab_size=32, dim=16, depth=2)
    embedded = torch.randn(2, 5, 16)

    out = model.forward_embedded(embedded)

    assert out.shape == (2, 5, 32)


def test_integrated_model_accepts_token_ids_on_text_path() -> None:
    model = EidosianSingularityModel(vocab_size=64, dim=16, depth=2)
    token_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    out = model(token_ids)

    assert out.shape == (2, 6, 64)


def test_integrated_model_projects_explicit_image_features() -> None:
    model = EidosianSingularityModel(vocab_size=64, dim=16, depth=2)
    image_features = torch.randn(2, 4, 512)

    out = model(image_features, modality=Modality.IMAGE)

    assert out.shape == (2, 4, 64)


def test_integrated_model_requires_explicit_modality_for_nonshared_features() -> None:
    model = EidosianSingularityModel(vocab_size=64, dim=16, depth=2)
    image_features = torch.randn(2, 4, 512)

    try:
        model(image_features)
    except ValueError as exc:
        assert "Explicit modality is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for untyped non-text features")
