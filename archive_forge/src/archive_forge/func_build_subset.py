import io
import math
import os
import typing
import weakref
def build_subset(buffer, unc_set, gid_set):
    """Build font subset using fontTools.

        Args:
            buffer: (bytes) the font given as a binary buffer.
            unc_set: (set) required glyph ids.
        Returns:
            Either None if subsetting is unsuccessful or the subset font buffer.
        """
    try:
        import fontTools.subset as fts
    except ImportError:
        if g_exceptions_verbose:
            fitz.exception_info()
        fitz.message('This method requires fontTools to be installed.')
        raise
    import tempfile
    tmp_dir = tempfile.gettempdir()
    oldfont_path = f'{tmp_dir}/oldfont.ttf'
    newfont_path = f'{tmp_dir}/newfont.ttf'
    uncfile_path = f'{tmp_dir}/uncfile.txt'
    args = [oldfont_path, '--retain-gids', f'--output-file={newfont_path}', "--layout-features='*'", '--passthrough-tables', '--ignore-missing-glyphs', '--ignore-missing-unicodes', '--symbol-cmap']
    with open(f'{tmp_dir}/uncfile.txt', 'w', encoding='utf8') as unc_file:
        if 65533 in unc_set:
            args.append(f'--gids-file={uncfile_path}')
            gid_set.add(189)
            unc_list = list(gid_set)
            for unc in unc_list:
                unc_file.write('%i\n' % unc)
        else:
            args.append(f'--unicodes-file={uncfile_path}')
            unc_set.add(255)
            unc_list = list(unc_set)
            for unc in unc_list:
                unc_file.write('%04x\n' % unc)
    with open(oldfont_path, 'wb') as fontfile:
        fontfile.write(buffer)
    try:
        os.remove(newfont_path)
    except Exception:
        pass
    try:
        fts.main(args)
        font = fitz.Font(fontfile=newfont_path)
        new_buffer = font.buffer
        if font.glyph_count == 0:
            new_buffer = None
    except Exception:
        fitz.exception_info()
        new_buffer = None
    try:
        os.remove(uncfile_path)
    except Exception:
        fitz.exception_info()
        pass
    try:
        os.remove(oldfont_path)
    except Exception:
        fitz.exception_info()
        pass
    try:
        os.remove(newfont_path)
    except Exception:
        fitz.exception_info()
        pass
    return new_buffer