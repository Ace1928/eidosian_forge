from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
def decode_attributed(text):
    """Create a document directly from some attributed text.

    See `pyglet.text.formats.attributed` for a description of attributed text.

    :Parameters:
        `text` : str
            Attributed text to decode.

    :rtype: `FormattedDocument`
    """
    decoder = get_decoder(None, 'text/vnd.pyglet-attributed')
    return decoder.decode(text)