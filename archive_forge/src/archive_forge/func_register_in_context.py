import json
import os
import readline
import sys
from yaql import __version__ as version
from yaql.language.exceptions import YaqlParsingException
from yaql.language import utils
def register_in_context(context, parser):
    context.register_function(lambda context, show_tokens: main(context, show_tokens, parser), name='__main')