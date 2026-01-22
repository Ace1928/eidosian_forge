from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from click import style
from black.output import err, out
def path_ignored(self, path: Path, message: str) -> None:
    if self.verbose:
        out(f'{path} ignored: {message}', bold=False)