from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
def types_table_metadata(dialect: str):
    from sqlalchemy import TEXT, Boolean, Column, DateTime, Float, Integer, MetaData, Table
    date_type = TEXT if dialect == 'sqlite' else DateTime
    bool_type = Integer if dialect == 'sqlite' else Boolean
    metadata = MetaData()
    types = Table('types', metadata, Column('TextCol', TEXT), Column('DateCol', date_type), Column('IntDateCol', Integer), Column('IntDateOnlyCol', Integer), Column('FloatCol', Float), Column('IntCol', Integer), Column('BoolCol', bool_type), Column('IntColWithNull', Integer), Column('BoolColWithNull', bool_type))
    return types