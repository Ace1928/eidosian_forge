import io
import math
import os
import typing
import weakref
def remove_hidden(cont_lines):
    """Remove hidden text from a PDF page.

        Args:
            cont_lines: list of lines with /Contents content. Should have status
                from after page.cleanContents().

        Returns:
            List of /Contents lines from which hidden text has been removed.

        Notes:
            The input must have been created after the page's /Contents object(s)
            have been cleaned with page.cleanContents(). This ensures a standard
            formatting: one command per line, single spaces between operators.
            This allows for drastic simplification of this code.
        """
    out_lines = []
    in_text = False
    suppress = False
    make_return = False
    for line in cont_lines:
        if line == b'BT':
            in_text = True
            out_lines.append(line)
            continue
        if line == b'ET':
            in_text = False
            out_lines.append(line)
            continue
        if line == b'3 Tr':
            suppress = True
            make_return = True
            continue
        if line[-2:] == b'Tr' and line[0] != b'3':
            suppress = False
            out_lines.append(line)
            continue
        if line == b'Q':
            suppress = False
            out_lines.append(line)
            continue
        if suppress and in_text:
            continue
        out_lines.append(line)
    if make_return:
        return out_lines
    else:
        return None