from typing import TYPE_CHECKING, Literal, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in
        picking a good starting learning rate.

        Args:
            model: Model to tune.
            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.
            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.
            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying val/test/predict
                samples used for running tuner on validation/testing/prediction.
            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
            method: Method to run tuner on. It can be any of ``("fit", "validate", "test", "predict")``.
            min_lr: minimum learning rate to investigate
            max_lr: maximum learning rate to investigate
            num_training: number of learning rates to test
            mode: Search strategy to update learning rate after each batch:

                - ``'exponential'``: Increases the learning rate exponentially.
                - ``'linear'``: Increases the learning rate linearly.

            early_stop_threshold: Threshold for stopping the search. If the
                loss at any point is larger than early_stop_threshold*best_loss
                then the search is stopped. To disable, set to None.
            update_attr: Whether to update the learning rate attribute or not.
            attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
                automatically detected. Otherwise, set the name here.

        Raises:
            MisconfigurationException:
                If learning rate/lr in ``model`` or ``model.hparams`` isn't overridden,
                or if you are using more than one optimizer.

        