from __future__ import annotations
import typing
import pandas as pd
from ..iapi import labels_view, panel_ranges, panel_view
from .coord_cartesian import coord_cartesian
def flip_labels(obj: THasLabels) -> THasLabels:
    """
    Rename fields x to y and y to x

    Parameters
    ----------
    obj : dict_like | dataclass
        Object with labels to rename
    """

    def sub(a: str, b: str, df: pd.DataFrame):
        """
        Substitute all keys that start with a to b
        """
        columns: list[str] = df.columns.tolist()
        for label in columns:
            if label.startswith(a):
                new_label = b + label[1:]
                df[new_label] = df.pop(label)
    if isinstance(obj, pd.DataFrame):
        sub('x', 'z', obj)
        sub('y', 'x', obj)
        sub('z', 'y', obj)
    elif isinstance(obj, (labels_view, panel_view)):
        obj.x, obj.y = (obj.y, obj.x)
    return obj