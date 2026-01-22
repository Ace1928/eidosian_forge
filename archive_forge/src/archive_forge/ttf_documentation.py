import os
import mmap
import struct
import codecs
Close the font file.

        This is a good idea, since the entire file is memory mapped in
        until this method is called.  After closing cannot rely on the
        ``get_*`` methods.
        