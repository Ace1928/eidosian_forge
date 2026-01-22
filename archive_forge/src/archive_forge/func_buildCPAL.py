import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def buildCPAL(palettes: Sequence[Sequence[Tuple[float, float, float, float]]], paletteTypes: Optional[Sequence[ColorPaletteType]]=None, paletteLabels: Optional[Sequence[_OptionalLocalizedString]]=None, paletteEntryLabels: Optional[Sequence[_OptionalLocalizedString]]=None, nameTable: Optional[_n_a_m_e.table__n_a_m_e]=None) -> C_P_A_L_.table_C_P_A_L_:
    """Build CPAL table from list of color palettes.

    Args:
        palettes: list of lists of colors encoded as tuples of (R, G, B, A) floats
            in the range [0..1].
        paletteTypes: optional list of ColorPaletteType, one for each palette.
        paletteLabels: optional list of palette labels. Each lable can be either:
            None (no label), a string (for for default English labels), or a
            localized string (as a dict keyed with BCP47 language codes).
        paletteEntryLabels: optional list of palette entry labels, one for each
            palette entry (see paletteLabels).
        nameTable: optional name table where to store palette and palette entry
            labels. Required if either paletteLabels or paletteEntryLabels is set.

    Return:
        A new CPAL v0 or v1 table, if custom palette types or labels are specified.
    """
    if len({len(p) for p in palettes}) != 1:
        raise ColorLibError('color palettes have different lengths')
    if (paletteLabels or paletteEntryLabels) and (not nameTable):
        raise TypeError('nameTable is required if palette or palette entries have labels')
    cpal = C_P_A_L_.table_C_P_A_L_()
    cpal.numPaletteEntries = len(palettes[0])
    cpal.palettes = []
    for i, palette in enumerate(palettes):
        colors = []
        for j, color in enumerate(palette):
            if not isinstance(color, tuple) or len(color) != 4:
                raise ColorLibError(f'In palette[{i}][{j}]: expected (R, G, B, A) tuple, got {color!r}')
            if any((v > 1 or v < 0 for v in color)):
                raise ColorLibError(f'palette[{i}][{j}] has invalid out-of-range [0..1] color: {color!r}')
            red, green, blue, alpha = color
            colors.append(C_P_A_L_.Color(*(round(v * 255) for v in (blue, green, red, alpha))))
        cpal.palettes.append(colors)
    if any((v is not None for v in (paletteTypes, paletteLabels, paletteEntryLabels))):
        cpal.version = 1
        if paletteTypes is not None:
            if len(paletteTypes) != len(palettes):
                raise ColorLibError(f'Expected {len(palettes)} paletteTypes, got {len(paletteTypes)}')
            cpal.paletteTypes = [ColorPaletteType(t).value for t in paletteTypes]
        else:
            cpal.paletteTypes = [C_P_A_L_.table_C_P_A_L_.DEFAULT_PALETTE_TYPE] * len(palettes)
        if paletteLabels is not None:
            if len(paletteLabels) != len(palettes):
                raise ColorLibError(f'Expected {len(palettes)} paletteLabels, got {len(paletteLabels)}')
            cpal.paletteLabels = buildPaletteLabels(paletteLabels, nameTable)
        else:
            cpal.paletteLabels = [C_P_A_L_.table_C_P_A_L_.NO_NAME_ID] * len(palettes)
        if paletteEntryLabels is not None:
            if len(paletteEntryLabels) != cpal.numPaletteEntries:
                raise ColorLibError(f'Expected {cpal.numPaletteEntries} paletteEntryLabels, got {len(paletteEntryLabels)}')
            cpal.paletteEntryLabels = buildPaletteLabels(paletteEntryLabels, nameTable)
        else:
            cpal.paletteEntryLabels = [C_P_A_L_.table_C_P_A_L_.NO_NAME_ID] * cpal.numPaletteEntries
    else:
        cpal.version = 0
    return cpal