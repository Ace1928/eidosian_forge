from os.path import dirname as _dirname
from os.path import splitext as _splitext
import pyglet
from pyglet.text import layout, document, caret
def decode_html(text, location=None):
    """Create a document directly from some HTML formatted text.

    :Parameters:
        `text` : str
            HTML data to decode.
        `location` : str
            Location giving the base path for additional resources
            referenced from the document (e.g., images).

    :rtype: `FormattedDocument`
    """
    decoder = get_decoder(None, 'text/html')
    return decoder.decode(text, location)