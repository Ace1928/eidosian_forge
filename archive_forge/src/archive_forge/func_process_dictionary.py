from typing import Any, Mapping
import numpy as np
def process_dictionary(d: Mapping[str, Any]) -> Mapping[str, Any]:
    return {k: process_value(v) for k, v in d.items()}