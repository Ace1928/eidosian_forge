from .. import config
def iter_option_names(self):
    try:
        self._config.get((b'user',), b'email')
    except KeyError:
        pass
    else:
        yield 'email'
    try:
        self._config.get((b'user',), b'signingkey')
    except KeyError:
        pass
    else:
        yield 'gpg_signing_key'