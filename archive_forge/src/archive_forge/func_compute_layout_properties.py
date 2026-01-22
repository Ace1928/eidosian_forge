import calendar
import datetime as dt
import re
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from itertools import permutations
import bokeh
import numpy as np
import pandas as pd
from bokeh.core.json_encoder import serialize_json  # noqa (API import)
from bokeh.core.property.datetime import Datetime
from bokeh.core.validation import silence
from bokeh.layouts import Column, Row, group_tools
from bokeh.models import (
from bokeh.models.formatters import PrintfTickFormatter, TickFormatter
from bokeh.models.scales import CategoricalScale, LinearScale, LogScale
from bokeh.models.widgets import DataTable, Div
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.themes.theme import Theme
from packaging.version import Version
from ...core.layout import Layout
from ...core.ndmapping import NdMapping
from ...core.overlay import NdOverlay, Overlay
from ...core.spaces import DynamicMap, get_nested_dmaps
from ...core.util import (
from ...util.warnings import warn
from ..util import dim_axis_label
def compute_layout_properties(width, height, frame_width, frame_height, explicit_width, explicit_height, aspect, data_aspect, responsive, size_multiplier, logger=None):
    """
    Utility to compute the aspect, plot width/height and sizing_mode
    behavior.

    Args:
      width (int): Plot width
      height (int): Plot height
      frame_width (int): Plot frame width
      frame_height (int): Plot frame height
      explicit_width (list): List of user supplied widths
      explicit_height (list): List of user supplied heights
      aspect (float): Plot aspect
      data_aspect (float): Scaling between x-axis and y-axis ranges
      responsive (boolean): Whether the plot should resize responsively
      size_multiplier (float): Multiplier for supplied plot dimensions
      logger (param.Parameters): Parameters object to issue warnings on

    Returns:
      Returns two dictionaries one for the aspect and sizing modes,
      and another for the plot dimensions.
    """
    fixed_width = explicit_width or frame_width
    fixed_height = explicit_height or frame_height
    fixed_aspect = aspect or data_aspect
    if aspect == 'square':
        aspect = 1
    elif aspect == 'equal':
        data_aspect = 1
    height = None if height is None else int(height * size_multiplier)
    width = None if width is None else int(width * size_multiplier)
    frame_height = None if frame_height is None else int(frame_height * size_multiplier)
    frame_width = None if frame_width is None else int(frame_width * size_multiplier)
    actual_width = frame_width or width
    actual_height = frame_height or height
    if frame_width is not None:
        width = None
    if frame_height is not None:
        height = None
    sizing_mode = 'fixed'
    if responsive:
        if fixed_height and fixed_width:
            responsive = False
            if logger:
                logger.warning('responsive mode could not be enabled because fixed width and height were specified.')
        elif fixed_width:
            height = None
            sizing_mode = 'fixed' if fixed_aspect else 'stretch_height'
        elif fixed_height:
            width = None
            sizing_mode = 'fixed' if fixed_aspect else 'stretch_width'
        else:
            width, height = (None, None)
            if fixed_aspect:
                if responsive == 'width':
                    sizing_mode = 'scale_width'
                elif responsive == 'height':
                    sizing_mode = 'scale_height'
                else:
                    sizing_mode = 'scale_both'
            elif responsive == 'width':
                sizing_mode = 'stretch_both'
            elif responsive == 'height':
                sizing_mode = 'stretch_height'
            else:
                sizing_mode = 'stretch_both'
    if fixed_aspect:
        if (explicit_width and (not frame_width)) != (explicit_height and (not frame_height)) and logger:
            logger.warning('Due to internal constraints, when aspect and width/height is set, the bokeh backend uses those values as frame_width/frame_height instead. This ensures the aspect is respected, but means that the plot might be slightly larger than anticipated. Set the frame_width/frame_height explicitly to suppress this warning.')
        aspect_type = 'data_aspect' if data_aspect else 'aspect'
        if fixed_width and fixed_height and aspect:
            if aspect == 'equal':
                data_aspect = 1
            elif not data_aspect:
                aspect = None
                if logger:
                    logger.warning('%s value was ignored because absolute width and height values were provided. Either supply explicit frame_width and frame_height to achieve desired aspect OR supply a combination of width or height and an aspect value.' % aspect_type)
        elif fixed_width and responsive:
            height = None
            responsive = False
            if logger:
                logger.warning('responsive mode could not be enabled because fixed width and aspect were specified.')
        elif fixed_height and responsive:
            width = None
            responsive = False
            if logger:
                logger.warning('responsive mode could not be enabled because fixed height and aspect were specified.')
        elif responsive == 'width':
            sizing_mode = 'scale_width'
        elif responsive == 'height':
            sizing_mode = 'scale_height'
    if responsive == 'width' and fixed_width:
        responsive = False
        if logger:
            logger.warning('responsive width mode could not be enabled because a fixed width was defined.')
    if responsive == 'height' and fixed_height:
        responsive = False
        if logger:
            logger.warning('responsive height mode could not be enabled because a fixed height was defined.')
    match_aspect = False
    aspect_scale = 1
    aspect_ratio = None
    if data_aspect:
        match_aspect = True
        if fixed_width and fixed_height:
            frame_width, frame_height = (frame_width or width, frame_height or height)
        elif fixed_width or not fixed_height:
            height = None
        elif fixed_height or not fixed_width:
            width = None
        aspect_scale = data_aspect
        if aspect == 'equal':
            aspect_scale = 1
        elif responsive:
            aspect_ratio = aspect
    elif fixed_width and fixed_height:
        pass
    elif isnumeric(aspect):
        if responsive:
            aspect_ratio = aspect
        elif fixed_width:
            frame_width = actual_width
            frame_height = int(actual_width / aspect)
            width, height = (None, None)
        else:
            frame_width = int(actual_height * aspect)
            frame_height = actual_height
            width, height = (None, None)
    elif aspect is not None and logger:
        logger.warning("aspect value of type %s not recognized, provide a numeric value, 'equal' or 'square'.")
    aspect_info = {'aspect_ratio': aspect_ratio, 'aspect_scale': aspect_scale, 'match_aspect': match_aspect, 'sizing_mode': sizing_mode}
    dimension_info = {'frame_width': frame_width, 'frame_height': frame_height, 'height': height, 'width': width}
    return (aspect_info, dimension_info)