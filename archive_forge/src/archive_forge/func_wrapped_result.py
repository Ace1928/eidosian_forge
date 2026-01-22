from __future__ import unicode_literals
import re
def wrapped_result():
    if or_list == []:
        return wrap(result)
    else:
        or_list.append(result)
        return Any([wrap(i) for i in or_list])