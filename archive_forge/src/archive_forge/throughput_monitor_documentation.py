import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
import torch
from typing_extensions import override
from lightning_fabric.plugins import Precision as FabricPrecision
from lightning_fabric.utilities.throughput import Throughput, get_available_flops
from lightning_fabric.utilities.throughput import _plugin_to_compute_dtype as fabric_plugin_to_compute_dtype
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import (
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
Computes and logs throughput with the :class:`~lightning_fabric.utilities.throughput.Throughput`

    Example::

        class MyModel(LightningModule):
            def setup(self, stage):
                with torch.device("meta"):
                    model = MyModel()

                    def sample_forward():
                        batch = torch.randn(..., device="meta")
                        return model(batch)

                    self.flops_per_batch = measure_flops(model, sample_forward, loss_fn=torch.Tensor.sum)


        logger = ...
        throughput = ThroughputMonitor(batch_size_fn=lambda batch: batch.size(0))
        trainer = Trainer(max_steps=1000, log_every_n_steps=10, callbacks=throughput, logger=logger)
        model = MyModel()
        trainer.fit(model)

    Notes:
        - It assumes that the batch size is the same during all iterations.
        - It will try to access a ``flops_per_batch`` attribute on your ``LightningModule`` on every iteration.
          We suggest using the :func:`~lightning_fabric.utilities.throughput.measure_flops` function for this.
          You might want to compute it differently each time based on your setup.

    Args:
        batch_size_fn: A function to compute the number of samples given a batch.
        length_fn: A function to compute the number of items in a sample given a batch.
        \**kwargs: See available parameters in
            :class:`~lightning_fabric.utilities.throughput.Throughput`

    