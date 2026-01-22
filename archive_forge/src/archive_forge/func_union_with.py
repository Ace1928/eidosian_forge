import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
import cloudpickle
import numpy as np
import pandas as pd
from fugue import (
from triad import FileSystem, ParamDict, assert_or_throw, to_uuid
from tune._utils import from_base64, to_base64
from tune.concepts.flow import Trial
from tune.concepts.space import Space
from tune.constants import (
from tune.exceptions import TuneCompileError
def union_with(self, other: 'StudyResult') -> None:
    """Union with another result set and update itself

        :param other: the other result dataset

        .. note::
            This method also removes duplicated reports based on
            :meth:`tune.concepts.flow.trial.Trial.trial_id`. Each
            trial will have only the best report in the updated
            result
        """
    self._result = self._result.union(other._result).partition_by(TUNE_REPORT_ID, presort=TUNE_REPORT_METRIC).take(1).persist()