import os
import struct
from Xlib import X, error
Find an authentication entry matching FAMILY, ADDRESS and
        DISPNO.

        The name of the auth scheme must match one of the names in
        TYPES.  If several entries match, the first scheme in TYPES
        will be choosen.

        If an entry is found, the tuple (name, data) is returned,
        otherwise XNoAuthError is raised.
        