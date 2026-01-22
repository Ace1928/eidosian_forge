from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def write_multi_job_file(job_list: list[QCInput], filename: str):
    """
        Write a multijob file.

        Args:
            job_list (): List of jobs.
            filename (): Filename
        """
    with zopen(filename, mode='wt') as file:
        file.write(QCInput.multi_job_string(job_list))