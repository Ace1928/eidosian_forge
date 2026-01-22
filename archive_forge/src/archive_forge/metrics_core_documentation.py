import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
Add a metric to the metric family.

        Args:
          labels: A list of label values
          value: A dict of string state names to booleans
        