from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def parse_join_type(self, join_type: str) -> str:
    """Parse join type string to standard join type string

        :param join_type: the join type string
        :return: the standard join type string
        """
    join_type = join_type.replace(' ', '').replace('_', '').lower()
    if join_type in ['inner', 'cross']:
        return join_type
    if join_type in ['inner', 'join']:
        return 'inner'
    if join_type in ['semi', 'leftsemi']:
        return 'left_semi'
    if join_type in ['anti', 'leftanti']:
        return 'left_anti'
    if join_type in ['left', 'leftouter']:
        return 'left_outer'
    if join_type in ['right', 'rightouter']:
        return 'right_outer'
    if join_type in ['outer', 'full', 'fullouter']:
        return 'full_outer'
    raise NotImplementedError(join_type)