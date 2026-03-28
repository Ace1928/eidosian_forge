from __future__ import annotations

import torch

from neural_forge.models.integrated import EidosianSingularityModel
from neural_forge.training.trainer import SingularityTrainer


def test_singularity_trainer_train_step_returns_scalar_loss() -> None:
    model = EidosianSingularityModel(vocab_size=32, dim=16, depth=2)
    trainer = SingularityTrainer(model, lr=1e-3)
    x = torch.randint(0, 32, (2, 5), dtype=torch.long)
    y = torch.randint(0, 32, (2, 5), dtype=torch.long)

    loss = trainer.train_step(x, y)

    assert isinstance(loss, float)
    assert loss >= 0.0
