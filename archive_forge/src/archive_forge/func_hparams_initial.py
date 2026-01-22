import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
from lightning_fabric.utilities.data import AttributeDict
from pytorch_lightning.utilities.parsing import save_hyperparameters
@property
def hparams_initial(self) -> AttributeDict:
    """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.

        Returns:
            AttributeDict: immutable initial hyperparameters

        """
    if not hasattr(self, '_hparams_initial'):
        return AttributeDict()
    return copy.deepcopy(self._hparams_initial)