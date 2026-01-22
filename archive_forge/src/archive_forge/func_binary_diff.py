import base64
import re
from io import BytesIO
from .... import errors, registry
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....timestamp import format_highres_date, unpack_highres_date
def binary_diff(old_filename, old_lines, new_filename, new_lines, to_file):
    temp = BytesIO()
    internal_diff(old_filename, old_lines, new_filename, new_lines, temp, allow_binary=True)
    temp.seek(0)
    base64.encode(temp, to_file)
    to_file.write(b'\n')