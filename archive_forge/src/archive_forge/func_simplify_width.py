from __future__ import annotations
import dataclasses
import enum
import typing
def simplify_width(width_type: Literal['clip', 'pack', 'given', 'relative', 'weight'] | WHSettings, width_amount: int | float | None) -> Literal[WHSettings.CLIP, WHSettings.PACK] | int | tuple[Literal[WHSettings.RELATIVE], int] | tuple[Literal[WHSettings.WEIGHT], int | float]:
    """
    Recombine (width_type, width_amount) into an width value.
    Inverse of normalize_width.
    """
    if width_type in {WHSettings.CLIP, WHSettings.PACK}:
        return WHSettings(width_type)
    if not isinstance(width_amount, int):
        raise TypeError(width_amount)
    if width_type == WHSettings.GIVEN:
        return width_amount
    return (WHSettings(width_type), width_amount)