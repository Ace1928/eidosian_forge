from __future__ import annotations
import json
from typing import Iterable
def stylesheet(self) -> str:
    return f'https://fonts.googleapis.com/css2?family={self.name.replace(' ', '+')}:wght@{';'.join((str(weight) for weight in self.weights))}&display=swap'