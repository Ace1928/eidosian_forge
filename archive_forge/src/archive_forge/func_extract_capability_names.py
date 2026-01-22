from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def extract_capability_names(capabilities):
    return {parse_capability(c)[0] for c in capabilities}