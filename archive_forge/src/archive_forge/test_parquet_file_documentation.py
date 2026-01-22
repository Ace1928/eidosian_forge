import io
import os
import sys
import pytest
import pyarrow as pa

    Unopened files should be closed explicitly after use,
    and previously opened files should be left open.
    Applies to read_table, ParquetDataset, and ParquetFile
    