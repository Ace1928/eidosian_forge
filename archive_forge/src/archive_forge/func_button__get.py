import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def button__get(self):
    return self.become(Button)