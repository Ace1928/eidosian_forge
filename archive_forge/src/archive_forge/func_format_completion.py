import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
def format_completion(self, item: click.shell_completion.CompletionItem) -> str:
    return f'{item.value}:::{item.help or ' '}'