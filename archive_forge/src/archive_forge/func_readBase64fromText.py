import base64
import sys
def readBase64fromText(text):
    if sys.version_info[0] <= 2:
        return base64.b64decode(text)
    else:
        return base64.b64decode(text.encode())