import ast
import string
@staticmethod
def transcode(blob):
    if isinstance(blob, bytes):
        blob = blob.decode('latin-1')
    return blob