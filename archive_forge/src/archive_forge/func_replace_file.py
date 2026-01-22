import os
import os.path
import re
from debian.deprecation import function_deprecated_by
import debian._arch_table
def replace_file(lines, local, encoding='UTF-8'):
    local_new = local + '.new'
    try:
        with open(local_new, 'w+', encoding=encoding) as new_file:
            for l in lines:
                new_file.write(l)
        os.replace(local_new, local)
    finally:
        if os.path.exists(local_new):
            os.unlink(local_new)