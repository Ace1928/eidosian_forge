from functools import partial
import email.base64mime
import email.quoprimime
from email import errors
from email.encoders import encode_7or8bit
def header_encode_lines(self, string, maxlengths):
    """Header-encode a string by converting it first to bytes.

        This is similar to `header_encode()` except that the string is fit
        into maximum line lengths as given by the argument.

        :param string: A unicode string for the header.  It must be possible
            to encode this string to bytes using the character set's
            output codec.
        :param maxlengths: Maximum line length iterator.  Each element
            returned from this iterator will provide the next maximum line
            length.  This parameter is used as an argument to built-in next()
            and should never be exhausted.  The maximum line lengths should
            not count the RFC 2047 chrome.  These line lengths are only a
            hint; the splitter does the best it can.
        :return: Lines of encoded strings, each with RFC 2047 chrome.
        """
    codec = self.output_codec or 'us-ascii'
    header_bytes = _encode(string, codec)
    encoder_module = self._get_encoder(header_bytes)
    encoder = partial(encoder_module.header_encode, charset=codec)
    charset = self.get_output_charset()
    extra = len(charset) + RFC2047_CHROME_LEN
    lines = []
    current_line = []
    maxlen = next(maxlengths) - extra
    for character in string:
        current_line.append(character)
        this_line = EMPTYSTRING.join(current_line)
        length = encoder_module.header_length(_encode(this_line, charset))
        if length > maxlen:
            current_line.pop()
            if not lines and (not current_line):
                lines.append(None)
            else:
                separator = ' ' if lines else ''
                joined_line = EMPTYSTRING.join(current_line)
                header_bytes = _encode(joined_line, codec)
                lines.append(encoder(header_bytes))
            current_line = [character]
            maxlen = next(maxlengths) - extra
    joined_line = EMPTYSTRING.join(current_line)
    header_bytes = _encode(joined_line, codec)
    lines.append(encoder(header_bytes))
    return lines