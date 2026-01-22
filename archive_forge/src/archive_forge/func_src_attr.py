from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def src_attr(self):
    import base64
    if self.embed and self.data is not None:
        data = base64 = base64.b64encode(self.data).decode('ascii')
        return 'data:{type};base64,{base64}'.format(type=self.mimetype, base64=data)
    elif self.url is not None:
        return self.url
    else:
        return ''