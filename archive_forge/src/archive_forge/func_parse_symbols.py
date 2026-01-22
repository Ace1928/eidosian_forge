import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
def parse_symbols(self, input_file) -> None:
    if len(self.symbols) > 0:
        return
    output = subprocess.check_output(['grep', 'define', input_file]).decode().splitlines()
    for line in output:
        symbol = self._extract_symbol(line)
        if symbol is None:
            continue
        self._symbols[symbol.name] = symbol
    self._group_symbols()