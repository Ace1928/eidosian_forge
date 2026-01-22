from __future__ import annotations
import os
from dataclasses import dataclass, field
Set some computed values derived from main_script_path.

        The usage of object.__setattr__ is necessary because trying to set
        self.script_folder or self.name normally, even within the __init__ method, will
        explode since we declared this dataclass to be frozen.

        We do this in __post_init__ so that we can use the auto-generated __init__
        method that most dataclasses use.
        