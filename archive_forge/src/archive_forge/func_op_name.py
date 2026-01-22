import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
@property
def op_name(self) -> str:
    return self._op_name