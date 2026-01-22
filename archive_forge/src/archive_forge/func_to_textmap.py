import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def to_textmap(self, layout: bool=False, layout_width=0, layout_height=0, layout_width_chars: int=0, layout_height_chars: int=0, x_density=DEFAULT_X_DENSITY, y_density=DEFAULT_Y_DENSITY, x_shift=0, y_shift=0, y_tolerance=DEFAULT_Y_TOLERANCE, use_text_flow: bool=False, presorted: bool=False, expand_ligatures: bool=True) -> TextMap:
    """
        Given a list of (word, chars) tuples (i.e., a WordMap), return a list of
        (char-text, char) tuples (i.e., a TextMap) that can be used to mimic the
        structural layout of the text on the page(s), using the following approach:

        - Sort the words by (doctop, x0) if not already sorted.

        - Calculate the initial doctop for the starting page.

        - Cluster the words by doctop (taking `y_tolerance` into account), and
          iterate through them.

        - For each cluster, calculate the distance between that doctop and the
          initial doctop, in points, minus `y_shift`. Divide that distance by
          `y_density` to calculate the minimum number of newlines that should come
          before this cluster. Append that number of newlines *minus* the number of
          newlines already appended, with a minimum of one.

        - Then for each cluster, iterate through each word in it. Divide each
          word's x0, minus `x_shift`, by `x_density` to calculate the minimum
          number of characters that should come before this cluster.  Append that
          number of spaces *minus* the number of characters and spaces already
          appended, with a minimum of one. Then append the word's text.

        - At the termination of each line, add more spaces if necessary to
          mimic `layout_width`.

        - Finally, add newlines to the end if necessary to mimic to
          `layout_height`.

        Note: This approach currently works best for horizontal, left-to-right
        text, but will display all words regardless of orientation. There is room
        for improvement in better supporting right-to-left text, as well as
        vertical text.
        """
    _textmap = []
    if not len(self.tuples):
        return TextMap(_textmap)
    expansions = LIGATURES if expand_ligatures else {}
    if layout:
        if layout_width_chars:
            if layout_width:
                raise ValueError('`layout_width` and `layout_width_chars` cannot both be set.')
        else:
            layout_width_chars = int(round(layout_width / x_density))
        if layout_height_chars:
            if layout_height:
                raise ValueError('`layout_height` and `layout_height_chars` cannot both be set.')
        else:
            layout_height_chars = int(round(layout_height / y_density))
        blank_line = [(' ', None)] * layout_width_chars
    else:
        blank_line = []
    num_newlines = 0
    words_sorted_doctop = self.tuples if presorted or use_text_flow else sorted(self.tuples, key=lambda x: float(x[0]['doctop']))
    first_word = words_sorted_doctop[0][0]
    doctop_start = first_word['doctop'] - first_word['top']
    for i, ws in enumerate(cluster_objects(words_sorted_doctop, lambda x: float(x[0]['doctop']), y_tolerance)):
        y_dist = (ws[0][0]['doctop'] - (doctop_start + y_shift)) / y_density if layout else 0
        num_newlines_prepend = max(int(i > 0), round(y_dist) - num_newlines)
        for i in range(num_newlines_prepend):
            if not len(_textmap) or _textmap[-1][0] == '\n':
                _textmap += blank_line
            _textmap.append(('\n', None))
        num_newlines += num_newlines_prepend
        line_len = 0
        line_words_sorted_x0 = ws if presorted or use_text_flow else sorted(ws, key=lambda x: float(x[0]['x0']))
        for word, chars in line_words_sorted_x0:
            x_dist = (word['x0'] - x_shift) / x_density if layout else 0
            num_spaces_prepend = max(min(1, line_len), round(x_dist) - line_len)
            _textmap += [(' ', None)] * num_spaces_prepend
            line_len += num_spaces_prepend
            for c in chars:
                letters = expansions.get(c['text'], c['text'])
                for letter in letters:
                    _textmap.append((letter, c))
                    line_len += 1
        if layout:
            _textmap += [(' ', None)] * (layout_width_chars - line_len)
    if layout:
        num_newlines_append = layout_height_chars - (num_newlines + 1)
        for i in range(num_newlines_append):
            if i > 0:
                _textmap += blank_line
            _textmap.append(('\n', None))
        if _textmap[-1] == ('\n', None):
            _textmap = _textmap[:-1]
    return TextMap(_textmap)