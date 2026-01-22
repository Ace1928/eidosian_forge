import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def check_reader(self, reader, expected_schema, expected_data):
    assert reader.schema == expected_schema
    batches = list(reader)
    assert len(batches) == len(expected_data)
    for batch, expected_batch in zip(batches, expected_data):
        batch.validate(full=True)
        assert batch.schema == expected_schema
        assert batch.to_pydict() == expected_batch