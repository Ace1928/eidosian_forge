import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def get_renames(self) -> Iterable[Tuple[Path, Path]]:
    """
        Files can be renamed in a refactoring.
        """
    return sorted(self._renames)