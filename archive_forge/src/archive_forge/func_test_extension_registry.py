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
def test_extension_registry(self):

    def my_creator() -> pd.DataFrame:
        return pd.DataFrame([[0, 1], [1, 2]], columns=['a', 'b'])

    def my_processor(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def my_transformer(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def my_out_transformer(df: pd.DataFrame) -> None:
        print(df)

    def my_outputter(df: pd.DataFrame) -> None:
        print(df)
    register_creator('mc', my_creator)
    register_processor('mp', my_processor)
    register_transformer('mt', my_transformer)
    register_output_transformer('mot', my_out_transformer)
    register_outputter('mo', my_outputter)
    with FugueWorkflow() as dag:
        df = dag.create('mc').process('mp').transform('mt')
        df.out_transform('mot')
        df.output('mo')
    dag.run(self.engine)