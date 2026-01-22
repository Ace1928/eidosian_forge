from abc import ABC, abstractmethod
from typing import (Dict, Any, Sequence, TextIO, Iterator, Optional, Union,
import re
from warnings import warn
from pathlib import Path, PurePath
import numpy as np
import ase
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import ParseError, read
from ase.io.utils import ImageChunk
from ase.calculators.singlepoint import SinglePointDFTCalculator, SinglePointKPoint
def outcarchunks(fd: TextIO, chunk_parser: ChunkParser=None, header_parser: HeaderParser=None) -> Iterator[OUTCARChunk]:
    """Function to build chunks of OUTCAR from a file stream"""
    name = Path(fd.name)
    workdir = name.parent
    header_parser = header_parser or OutcarHeaderParser(workdir=workdir)
    lines = build_header(fd)
    header = header_parser.build(lines)
    assert isinstance(header, dict)
    chunk_parser = chunk_parser or OutcarChunkParser()
    while True:
        try:
            lines = build_chunk(fd)
        except StopIteration:
            return
        yield OUTCARChunk(lines, header, parser=chunk_parser)