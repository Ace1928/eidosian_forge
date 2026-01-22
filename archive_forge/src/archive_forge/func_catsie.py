import contextlib
import dataclasses
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Generic, Iterable, List, Optional, TypeVar, Union
import catalogue
import confection
@my_registry.cats('catsie.v3')
def catsie(arg: Cat) -> Cat:
    return arg