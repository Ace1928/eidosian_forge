import threading
import numpy as np
from ..core import Format
The ClipboardGrabFormat provided a means to grab image data from
    the clipboard, using the uri "<clipboard>"

    This functionality is provided via Pillow. Note that "<clipboard>" is
    only supported on Windows.

    Parameters for reading
    ----------------------
    No parameters.
    