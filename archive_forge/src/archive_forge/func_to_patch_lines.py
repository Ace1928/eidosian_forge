import re
from .. import osutils
from ..iterablefile import IterableFile
def to_patch_lines(stanza, max_width=72):
    """Convert a stanza into RIO-Patch format lines.

    RIO-Patch is a RIO variant designed to be e-mailed as part of a patch.
    It resists common forms of damage such as newline conversion or the removal
    of trailing whitespace, yet is also reasonably easy to read.

    :param max_width: The maximum number of characters per physical line.
    :return: a list of lines
    """
    if max_width <= 6:
        raise ValueError(max_width)
    max_rio_width = max_width - 4
    lines = []
    for pline in stanza.to_lines():
        for line in pline.split(b'\n')[:-1]:
            line = re.sub(b'\\\\', b'\\\\\\\\', line)
            while len(line) > 0:
                partline = line[:max_rio_width]
                line = line[max_rio_width:]
                if len(line) > 0 and line[:1] != [b' ']:
                    break_index = -1
                    break_index = partline.rfind(b' ', -20)
                    if break_index < 3:
                        break_index = partline.rfind(b'-', -20)
                        break_index += 1
                    if break_index < 3:
                        break_index = partline.rfind(b'/', -20)
                    if break_index >= 3:
                        line = partline[break_index:] + line
                        partline = partline[:break_index]
                if len(line) > 0:
                    line = b'  ' + line
                partline = re.sub(b'\r', b'\\\\r', partline)
                blank_line = False
                if len(line) > 0:
                    partline += b'\\'
                elif re.search(b' $', partline):
                    partline += b'\\'
                    blank_line = True
                lines.append(b'# ' + partline + b'\n')
                if blank_line:
                    lines.append(b'#   \n')
    return lines