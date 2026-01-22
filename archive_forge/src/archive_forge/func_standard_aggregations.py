from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
from mlflow.utils.annotations import experimental
from mlflow.utils.validation import _is_numeric
def standard_aggregations(scores):
    return {'mean': np.mean(scores), 'variance': np.var(scores), 'p90': np.percentile(scores, 90)}