import re
from abc import ABC, abstractmethod
from typing import List, Union
from .text import Span, Text
Highlight :class:`rich.text.Text` using regular expressions.

        Args:
            text (~Text): Text to highlighted.

        