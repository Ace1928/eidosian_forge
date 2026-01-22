from os import getenv
from warnings import warn
from requests import Session
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
from .utils_worker import MonoWorker
@property
def message_id(self):
    if hasattr(self, '_message_id'):
        return self._message_id
    try:
        res = self.session.post(self.API + '%s/sendMessage' % self.token, data={'text': '`' + self.text + '`', 'chat_id': self.chat_id, 'parse_mode': 'MarkdownV2'}).json()
    except Exception as e:
        tqdm_auto.write(str(e))
    else:
        if res.get('error_code') == 429:
            warn('Creation rate limit: try increasing `mininterval`.', TqdmWarning, stacklevel=2)
        else:
            self._message_id = res['result']['message_id']
            return self._message_id