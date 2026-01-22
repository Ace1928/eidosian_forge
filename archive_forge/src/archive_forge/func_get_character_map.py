import os
import mmap
import struct
import codecs
def get_character_map(self):
    """Return the character map.

        Returns a dictionary where the key is a unit-length unicode
        string and the value is a glyph index.  Currently only 
        format 4 character maps are read.
        """
    if self._character_map:
        return self._character_map
    cmap = _read_cmap_header(self._data, self._tables['cmap'].offset)
    records = _read_cmap_encoding_record.array(self._data, self._tables['cmap'].offset + cmap.size, cmap.num_tables)
    self._character_map = {}
    for record in records:
        if record.platform_id == 3 and record.encoding_id == 1:
            offset = self._tables['cmap'].offset + record.offset
            format_header = _read_cmap_format_header(self._data, offset)
            if format_header.format == 4:
                self._character_map = self._get_character_map_format4(offset)
                break
    return self._character_map