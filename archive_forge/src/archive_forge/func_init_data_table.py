import abc
from typing import Any, Dict, List, Optional
from tensorflow.keras.callbacks import Callback  # type: ignore
import wandb
from wandb.sdk.lib import telemetry
def init_data_table(self, column_names: List[str]) -> None:
    """Initialize the W&B Tables for validation data.

        Call this method `on_train_begin` or equivalent hook. This is followed by adding
        data to the table row or column wise.

        Args:
            column_names: (list) Column names for W&B Tables.
        """
    self.data_table = wandb.Table(columns=column_names, allow_mixed_types=True)