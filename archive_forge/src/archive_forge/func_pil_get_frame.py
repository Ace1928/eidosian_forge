import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
def pil_get_frame(im, is_gray=None, as_gray=None, mode=None, dtype=None):
    """
    is_gray: Whether the image *is* gray (by inspecting its palette).
    as_gray: Whether the resulting image must be converted to gaey.
    mode: The mode to convert to.
    """
    if is_gray is None:
        is_gray = _palette_is_grayscale(im)
    frame = im
    if mode is not None:
        if mode != im.mode:
            frame = im.convert(mode)
    elif as_gray:
        pass
    elif im.mode == 'P' and is_gray:
        frame = im.convert('L')
    elif im.mode == 'P':
        if im.info.get('transparency', None) is not None:
            frame = im.convert('RGBA')
        elif im.palette.mode in ('RGB', 'RGBA'):
            p = np.frombuffer(im.palette.getdata()[1], np.uint8)
            if hasattr(im.palette, 'rawmode_saved'):
                im.palette.rawmode = im.palette.rawmode_saved
            mode = im.palette.rawmode if im.palette.rawmode else im.palette.mode
            nchannels = len(mode)
            p.shape = (-1, nchannels)
            if p.shape[1] == 3 or (p.shape[1] == 4 and mode[-1] == 'X'):
                p = np.column_stack((p[:, :3], 255 * np.ones(p.shape[0], p.dtype)))
            if mode.startswith('BGR'):
                p = p[:, [2, 1, 0]] if p.shape[1] == 3 else p[:, [2, 1, 0, 3]]
            frame_paletted = np.array(im, np.uint8)
            try:
                frame = p[frame_paletted]
            except Exception:
                frame = im.convert('RGBA')
        elif True:
            frame = im.convert('RGBA')
        else:
            frame = im.convert('RGB')
    elif 'A' in im.mode:
        frame = im.convert('RGBA')
    elif im.mode == 'CMYK':
        frame = im.convert('RGB')
    elif im.format == 'GIF' and im.mode == 'RGB':
        frame = im.convert('RGBA')
    if as_gray:
        frame = frame.convert('F')
    elif not isinstance(frame, np.ndarray) and frame.mode == '1':
        frame = frame.convert('L')
    if im.mode.startswith('I;16'):
        shape = im.size
        dtype = '>u2' if im.mode.endswith('B') else '<u2'
        if 'S' in im.mode:
            dtype = dtype.replace('u', 'i')
        frame = np.frombuffer(frame.tobytes(), dtype).copy()
        frame.shape = shape[::-1]
    else:
        if im.format == 'PNG' and im.mode == 'I' and (dtype is None):
            dtype = 'uint16'
        frame = np.array(frame, dtype=dtype)
    return frame