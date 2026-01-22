import contextlib
import dataclasses
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Generic, Iterable, List, Optional, TypeVar, Union
import catalogue
import confection
@my_registry.cats('generic_cat.v1')
def generic_cat(cat: Cat[int, int]) -> Cat[int, int]:
    cat.name = 'generic_cat'
    return cat