import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import srsly
import tqdm
from wasabi import Printer
from .. import util
from ..errors import Errors
from ..util import registry
The ConsoleLogger.v3 prints out training logs in the console and/or saves them to a jsonl file.
    progress_bar (Optional[str]): Type of progress bar to show in the console. Allowed values:
        train - Tracks the number of steps from the beginning of training until the full training run is complete (training.max_steps is reached).
        eval - Tracks the number of steps between the previous and next evaluation (training.eval_frequency is reached).
    console_output (bool): Whether the logger should print the logs on the console.
    output_file (Optional[Union[str, Path]]): The file to save the training logs to.
    