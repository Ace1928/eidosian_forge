from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def load_seek_table(self, fp, seek_to_0):
    fsize = fp.seek(0, 2)
    if fsize == 0:
        return
    elif fsize < 17:
        msg = 'File size is less than the minimal size (17 bytes) of Zstandard Seekable Format.'
        raise SeekableFormatError(msg)
    fp.seek(-9, 2)
    footer = fp.read(9)
    frames_number, descriptor, magic_number = self._s_footer.unpack(footer)
    if magic_number != 2408770225:
        msg = 'The last 4 bytes of the file is not Zstandard Seekable Format Magic Number (b"\\xb1\\xea\\x92\\x8f)". SeekableZstdFile class only supports Zstandard Seekable Format file or 0-size file. To read a zstd file that is not in Zstandard Seekable Format, use ZstdFile class.'
        raise SeekableFormatError(msg)
    self._has_checksum = descriptor & 128
    if descriptor & 124:
        msg = 'In Zstandard Seekable Format version %s, the Reserved_Bits in Seek_Table_Descriptor must be 0.' % __format_version__
        raise SeekableFormatError(msg)
    entry_size = 12 if self._has_checksum else 8
    skippable_frame_size = 17 + frames_number * entry_size
    if fsize < skippable_frame_size:
        raise SeekableFormatError('File size is less than expected size of the seek table frame.')
    fp.seek(-skippable_frame_size, 2)
    skippable_frame = fp.read(skippable_frame_size)
    skippable_magic_number, content_size = self._s_2uint32.unpack_from(skippable_frame, 0)
    if skippable_magic_number != 407710302:
        msg = "Seek table frame's Magic_Number is wrong."
        raise SeekableFormatError(msg)
    if content_size != skippable_frame_size - 8:
        msg = "Seek table frame's Frame_Size is wrong."
        raise SeekableFormatError(msg)
    if seek_to_0:
        fp.seek(0)
    offset = 8
    for idx in range(frames_number):
        if self._has_checksum:
            compressed_size, decompressed_size, checksum = self._s_3uint32.unpack_from(skippable_frame, offset)
            offset += 12
        else:
            compressed_size, decompressed_size = self._s_2uint32.unpack_from(skippable_frame, offset)
            offset += 8
        if compressed_size == 0 and decompressed_size != 0:
            msg = 'Wrong seek table. The index %d frame (0-based) is 0 size, but decompressed size is non-zero, this is impossible.' % idx
            raise SeekableFormatError(msg)
        self.append_entry(compressed_size, decompressed_size)
        if self._full_c_size > fsize - skippable_frame_size:
            msg = 'Wrong seek table. Since index %d frame (0-based), the cumulated compressed size is greater than file size.' % idx
            raise SeekableFormatError(msg)
    if self._full_c_size != fsize - skippable_frame_size:
        raise SeekableFormatError('The cumulated compressed size is wrong')
    self._seek_frame_size = skippable_frame_size
    self._file_size = fsize