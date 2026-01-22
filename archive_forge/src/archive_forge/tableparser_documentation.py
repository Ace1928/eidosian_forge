import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars

        Check for text in column margins and text overflow in the last column.
        Raise TableMarkupError if anything but whitespace is in column margins.
        Adjust the end value for the last column if there is text overflow.
        