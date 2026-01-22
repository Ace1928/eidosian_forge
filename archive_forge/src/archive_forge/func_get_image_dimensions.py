import struct
import zlib
from django.core.files import File
def get_image_dimensions(file_or_path, close=False):
    """
    Return the (width, height) of an image, given an open file or a path.  Set
    'close' to True to close the file at the end if it is initially in an open
    state.
    """
    from PIL import ImageFile as PillowImageFile
    p = PillowImageFile.Parser()
    if hasattr(file_or_path, 'read'):
        file = file_or_path
        file_pos = file.tell()
        file.seek(0)
    else:
        try:
            file = open(file_or_path, 'rb')
        except OSError:
            return (None, None)
        close = True
    try:
        chunk_size = 1024
        while 1:
            data = file.read(chunk_size)
            if not data:
                break
            try:
                p.feed(data)
            except zlib.error as e:
                if e.args[0].startswith('Error -5'):
                    pass
                else:
                    raise
            except struct.error:
                pass
            except RuntimeError:
                pass
            if p.image:
                return p.image.size
            chunk_size *= 2
        return (None, None)
    finally:
        if close:
            file.close()
        else:
            file.seek(file_pos)