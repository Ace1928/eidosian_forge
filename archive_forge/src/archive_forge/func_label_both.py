from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
def label_both(label_info: strip_label_details, multi_line: bool=True, sep: str=': ') -> strip_label_details:
    """
    Concatenate the facet variable with the value

    Parameters
    ----------
    label_info : strip_label_details
        Label information to be modified.
    multi_line : bool
        Whether to place each variable on a separate line
    sep : str
        Separation between variable name and value

    Returns
    -------
    out : strip_label_details
        Label information
    """
    label_info = label_info.copy()
    for var, lvalue in label_info.variables.items():
        label_info.variables[var] = f'{var}{sep}{lvalue}'
    if not multi_line:
        label_info = label_info.collapse()
    return label_info