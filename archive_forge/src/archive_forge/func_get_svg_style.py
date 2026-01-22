import inspect
import os
import platform
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from getpass import getpass
from html import escape
from inspect import isclass
from itertools import islice
from math import ceil
from time import monotonic
from types import FrameType, ModuleType, TracebackType
from typing import (
from pip._vendor.rich._null_file import NULL_FILE
from . import errors, themes
from ._emoji_replace import _emoji_replace
from ._export_format import CONSOLE_HTML_FORMAT, CONSOLE_SVG_FORMAT
from ._fileno import get_fileno
from ._log_render import FormatTimeCallable, LogRender
from .align import Align, AlignMethod
from .color import ColorSystem, blend_rgb
from .control import Control
from .emoji import EmojiVariant
from .highlighter import NullHighlighter, ReprHighlighter
from .markup import render as render_markup
from .measure import Measurement, measure_renderables
from .pager import Pager, SystemPager
from .pretty import Pretty, is_expandable
from .protocol import rich_cast
from .region import Region
from .scope import render_scope
from .screen import Screen
from .segment import Segment
from .style import Style, StyleType
from .styled import Styled
from .terminal_theme import DEFAULT_TERMINAL_THEME, SVG_EXPORT_THEME, TerminalTheme
from .text import Text, TextType
from .theme import Theme, ThemeStack
def get_svg_style(style: Style) -> str:
    """Convert a Style to CSS rules for SVG."""
    if style in style_cache:
        return style_cache[style]
    css_rules = []
    color = _theme.foreground_color if style.color is None or style.color.is_default else style.color.get_truecolor(_theme)
    bgcolor = _theme.background_color if style.bgcolor is None or style.bgcolor.is_default else style.bgcolor.get_truecolor(_theme)
    if style.reverse:
        color, bgcolor = (bgcolor, color)
    if style.dim:
        color = blend_rgb(color, bgcolor, 0.4)
    css_rules.append(f'fill: {color.hex}')
    if style.bold:
        css_rules.append('font-weight: bold')
    if style.italic:
        css_rules.append('font-style: italic;')
    if style.underline:
        css_rules.append('text-decoration: underline;')
    if style.strike:
        css_rules.append('text-decoration: line-through;')
    css = ';'.join(css_rules)
    style_cache[style] = css
    return css