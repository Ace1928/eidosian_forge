import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def reserved_word(token, word):
    return token and token.type == TOKEN.RESERVED and (token.text == word)