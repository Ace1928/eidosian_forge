from mimetypes import guess_type
from . import base
from git.types import Literal
@property
def mime_type(self) -> str:
    """
        :return: String describing the mime type of this file (based on the filename)

        :note: Defaults to 'text/plain' in case the actual file type is unknown.
        """
    guesses = None
    if self.path:
        guesses = guess_type(str(self.path))
    return guesses and guesses[0] or self.DEFAULT_MIME_TYPE