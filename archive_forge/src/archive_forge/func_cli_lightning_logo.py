import logging
import os
from lightning_utilities import module_available
from lightning_fabric.utilities.seed import seed_everything  # noqa: E402
from lightning_fabric.utilities.warnings import disable_possible_user_warnings  # noqa: E402
from pytorch_lightning.callbacks import Callback  # noqa: E402
from pytorch_lightning.core import LightningDataModule, LightningModule  # noqa: E402
from pytorch_lightning.trainer import Trainer  # noqa: E402
import pytorch_lightning._graveyard  # noqa: E402, F401  # isort: skip
def cli_lightning_logo() -> None:
    print()
    print('\x1b[0;35m' + LIGHTNING_LOGO + '\x1b[0m')
    print()