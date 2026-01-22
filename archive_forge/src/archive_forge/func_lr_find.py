from typing import TYPE_CHECKING, Literal, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
def lr_find(self, model: 'pl.LightningModule', train_dataloaders: Optional[Union[TRAIN_DATALOADERS, 'pl.LightningDataModule']]=None, val_dataloaders: Optional[EVAL_DATALOADERS]=None, dataloaders: Optional[EVAL_DATALOADERS]=None, datamodule: Optional['pl.LightningDataModule']=None, method: Literal['fit', 'validate', 'test', 'predict']='fit', min_lr: float=1e-08, max_lr: float=1, num_training: int=100, mode: str='exponential', early_stop_threshold: Optional[float]=4.0, update_attr: bool=True, attr_name: str='') -> Optional['_LRFinder']:
    """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in
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

        """
    if method != 'fit':
        raise MisconfigurationException("method='fit' is the only valid configuration to run lr finder.")
    _check_tuner_configuration(train_dataloaders, val_dataloaders, dataloaders, method)
    _check_lr_find_configuration(self._trainer)
    from pytorch_lightning.callbacks.lr_finder import LearningRateFinder
    lr_finder_callback: Callback = LearningRateFinder(min_lr=min_lr, max_lr=max_lr, num_training_steps=num_training, mode=mode, early_stop_threshold=early_stop_threshold, update_attr=update_attr, attr_name=attr_name)
    lr_finder_callback._early_exit = True
    self._trainer.callbacks = [lr_finder_callback] + self._trainer.callbacks
    self._trainer.fit(model, train_dataloaders, val_dataloaders, datamodule)
    self._trainer.callbacks = [cb for cb in self._trainer.callbacks if cb is not lr_finder_callback]
    return lr_finder_callback.optimal_lr