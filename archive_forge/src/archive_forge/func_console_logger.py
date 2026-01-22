import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import srsly
import tqdm
from wasabi import Printer
from .. import util
from ..errors import Errors
from ..util import registry
@registry.loggers('spacy.ConsoleLogger.v2')
def console_logger(progress_bar: bool=False, console_output: bool=True, output_file: Optional[Union[str, Path]]=None):
    """The ConsoleLogger.v2 prints out training logs in the console and/or saves them to a jsonl file.
    progress_bar (bool): Whether the logger should print a progress bar tracking the steps till the next evaluation pass.
    console_output (bool): Whether the logger should print the logs on the console.
    output_file (Optional[Union[str, Path]]): The file to save the training logs to.
    """
    return console_logger_v3(progress_bar=None if progress_bar is False else 'eval', console_output=console_output, output_file=output_file)