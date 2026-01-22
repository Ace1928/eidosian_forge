import tokenize
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
def normalize_token_spacing(code):
    tokens = [(t[0], t[1]) for t in tokenize.generate_tokens(StringIO(code).readline)]
    return pretty_untokenize(tokens)