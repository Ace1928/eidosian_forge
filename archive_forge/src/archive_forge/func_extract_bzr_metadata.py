from io import BytesIO
from .. import osutils
def extract_bzr_metadata(message):
    """Extract Bazaar metadata from a commit message.

    :param message: Commit message to extract from
    :return: Tuple with original commit message and metadata object
    """
    split = message.split(b'\n--BZR--\n', 1)
    if len(split) != 2:
        return (message, None)
    return (split[0], parse_roundtripping_metadata(split[1]))