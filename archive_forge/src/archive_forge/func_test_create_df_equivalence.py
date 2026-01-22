import datetime
import os
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional
from unittest import TestCase
from uuid import uuid4
from triad.utils.io import write_text, join
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from pytest import raises
from triad import SerializableRLock
import fugue.api as fa
from fugue import (
from fugue.column import col
from fugue.column import functions as ff
from fugue.column import lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import (
def test_create_df_equivalence(self):
    ndf = fa.as_fugue_engine_df(self.engine, pd.DataFrame([[0]], columns=['a']))
    dag1 = FugueWorkflow()
    dag1.df(ndf).show()
    dag2 = FugueWorkflow()
    dag2.create(ndf).show()
    assert dag1.spec_uuid() == dag2.spec_uuid()