from __future__ import annotations
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
@property
@abstractmethod
def user_music_dir(self) -> str:
    """:return: music directory tied to the user"""