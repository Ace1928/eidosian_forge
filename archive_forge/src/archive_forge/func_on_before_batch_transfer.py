from typing import Any, Dict, Optional
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    """Override to alter or apply batch augmentations to your batch before it is transferred to the device.

        Note:
            To check the current state of execution of this hook you can use
            ``self.trainer.training/testing/validating/predicting`` so that you can
            add different logic as per your requirement.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data

        Example::

            def on_before_batch_transfer(self, batch, dataloader_idx):
                batch['x'] = transforms(batch['x'])
                return batch

        See Also:
            - :meth:`on_after_batch_transfer`
            - :meth:`transfer_batch_to_device`

        """
    return batch