import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def js_popup__get(self):
    return self.become(JSPopup)