from typing import Collection, Sequence, Tuple, Union
import abc
import dataclasses
import enum
import numpy as np
Construct a `RunTagFilter`.

        A time series passes this filter if both its run *and* its tag are
        included in the corresponding whitelists.

        Order and multiplicity are ignored; `runs` and `tags` are treated as
        sets.

        Args:
          runs: Collection of run names, as strings, or `None` to admit all
            runs.
          tags: Collection of tag names, as strings, or `None` to admit all
            tags.
        