import io
import math
import os
import typing
import weakref
def subset_fonts(doc: fitz.Document, verbose: bool=False, fallback: bool=False) -> None:
    """Build font subsets of a PDF. Requires package 'fontTools'.

    Eligible fonts are potentially replaced by smaller versions. Page text is
    NOT rewritten and thus should retain properties like being hidden or
    controlled by optional content.

    This method by default uses MuPDF's own internal feature to create subset
    fonts. As this is a new function, errors may still occur. In this case,
    please fall back to using the previous version by using "fallback=True".
    """
    if fallback is False:
        pdf = mupdf.pdf_document_from_fz_document(doc)
        mupdf.pdf_subset_fonts2(pdf, list(range(doc.page_count)))
        return
    font_buffers = {}

    def get_old_widths(xref):
        """Retrieve old font '/W' and '/DW' values."""
        df = doc.xref_get_key(xref, 'DescendantFonts')
        if df[0] != 'array':
            return (None, None)
        df_xref = int(df[1][1:-1].replace('0 R', ''))
        widths = doc.xref_get_key(df_xref, 'W')
        if widths[0] != 'array':
            widths = None
        else:
            widths = widths[1]
        dwidths = doc.xref_get_key(df_xref, 'DW')
        if dwidths[0] != 'int':
            dwidths = None
        else:
            dwidths = dwidths[1]
        return (widths, dwidths)

    def set_old_widths(xref, widths, dwidths):
        """Restore the old '/W' and '/DW' in subsetted font.

        If either parameter is None or evaluates to False, the corresponding
        dictionary key will be set to null.
        """
        df = doc.xref_get_key(xref, 'DescendantFonts')
        if df[0] != 'array':
            return None
        df_xref = int(df[1][1:-1].replace('0 R', ''))
        if (type(widths) is not str or not widths) and doc.xref_get_key(df_xref, 'W')[0] != 'null':
            doc.xref_set_key(df_xref, 'W', 'null')
        else:
            doc.xref_set_key(df_xref, 'W', widths)
        if (type(dwidths) is not str or not dwidths) and doc.xref_get_key(df_xref, 'DW')[0] != 'null':
            doc.xref_set_key(df_xref, 'DW', 'null')
        else:
            doc.xref_set_key(df_xref, 'DW', dwidths)
        return None

    def set_subset_fontname(new_xref):
        """Generate a name prefix to tag a font as subset.

        We use a random generator to select 6 upper case ASCII characters.
        The prefixed name must be put in the font xref as the "/BaseFont" value
        and in the FontDescriptor object as the '/FontName' value.
        """
        import random
        import string
        prefix = ''.join(random.choices(tuple(string.ascii_uppercase), k=6)) + '+'
        font_str = doc.xref_object(new_xref, compressed=True)
        font_str = font_str.replace('/BaseFont/', '/BaseFont/' + prefix)
        df = doc.xref_get_key(new_xref, 'DescendantFonts')
        if df[0] == 'array':
            df_xref = int(df[1][1:-1].replace('0 R', ''))
            fd = doc.xref_get_key(df_xref, 'FontDescriptor')
            if fd[0] == 'xref':
                fd_xref = int(fd[1].replace('0 R', ''))
                fd_str = doc.xref_object(fd_xref, compressed=True)
                fd_str = fd_str.replace('/FontName/', '/FontName/' + prefix)
                doc.update_object(fd_xref, fd_str)
        doc.update_object(new_xref, font_str)

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

    def repl_fontnames(doc):
        """Populate 'font_buffers'.

        For each font candidate, store its xref and the list of names
        by which PDF text may refer to it (there may be multiple).
        """

        def norm_name(name):
            """Recreate font name that contains PDF hex codes.

            E.g. #20 -> space, chr(32)
            """
            while '#' in name:
                p = name.find('#')
                c = int(name[p + 1:p + 3], 16)
                name = name.replace(name[p:p + 3], chr(c))
            return name

        def get_fontnames(doc, item):
            """Return a list of fontnames for an item of page.get_fonts().

            There may be multiple names e.g. for Type0 fonts.
            """
            fontname = item[3]
            names = [fontname]
            fontname = doc.xref_get_key(item[0], 'BaseFont')[1][1:]
            fontname = norm_name(fontname)
            if fontname not in names:
                names.append(fontname)
            descendents = doc.xref_get_key(item[0], 'DescendantFonts')
            if descendents[0] != 'array':
                return names
            descendents = descendents[1][1:-1]
            if descendents.endswith(' 0 R'):
                xref = int(descendents[:-4])
                descendents = doc.xref_object(xref, compressed=True)
            p1 = descendents.find('/BaseFont')
            if p1 >= 0:
                p2 = descendents.find('/', p1 + 1)
                p1 = min(descendents.find('/', p2 + 1), descendents.find('>>', p2 + 1))
                fontname = descendents[p2 + 1:p1]
                fontname = norm_name(fontname)
                if fontname not in names:
                    names.append(fontname)
            return names
        for i in range(doc.page_count):
            for f in doc.get_page_fonts(i, full=True):
                font_xref = f[0]
                font_ext = f[1]
                basename = f[3]
                if font_ext not in ('otf', 'ttf', 'woff', 'woff2'):
                    continue
                if len(basename) > 6 and basename[6] == '+':
                    continue
                extr = doc.extract_font(font_xref)
                fontbuffer = extr[-1]
                names = get_fontnames(doc, f)
                name_set, xref_set, subsets = font_buffers.get(fontbuffer, (set(), set(), (set(), set())))
                xref_set.add(font_xref)
                for name in names:
                    name_set.add(name)
                font = fitz.Font(fontbuffer=fontbuffer)
                name_set.add(font.name)
                del font
                font_buffers[fontbuffer] = (name_set, xref_set, subsets)

    def find_buffer_by_name(name):
        for buffer, (name_set, _, _) in font_buffers.items():
            if name in name_set:
                return buffer
        return None
    repl_fontnames(doc)
    if not font_buffers:
        if verbose:
            fitz.message(f'No fonts to subset.')
        return 0
    old_fontsize = 0
    new_fontsize = 0
    for fontbuffer in font_buffers.keys():
        old_fontsize += len(fontbuffer)
    for page in doc:
        for span in page.get_texttrace():
            if type(span) is not dict:
                continue
            fontname = span['font'][:33]
            buffer = find_buffer_by_name(fontname)
            if buffer is None:
                continue
            name_set, xref_set, (set_ucs, set_gid) = font_buffers[buffer]
            for c in span['chars']:
                set_ucs.add(c[0])
                set_gid.add(c[1])
            font_buffers[buffer] = (name_set, xref_set, (set_ucs, set_gid))
    for old_buffer, (name_set, xref_set, subsets) in font_buffers.items():
        new_buffer = build_subset(old_buffer, subsets[0], subsets[1])
        fontname = list(name_set)[0]
        if new_buffer is None or len(new_buffer) >= len(old_buffer):
            if verbose:
                fitz.message(f'Cannot subset {fontname!r}.')
            continue
        if verbose:
            fitz.message(f'Built subset of font {fontname!r}.')
        val = doc._insert_font(fontbuffer=new_buffer)
        new_xref = val[0]
        set_subset_fontname(new_xref)
        font_str = doc.xref_object(new_xref, compressed=True)
        for font_xref in xref_set:
            width_table, def_width = get_old_widths(font_xref)
            doc.update_object(font_xref, font_str)
            if width_table or def_width:
                set_old_widths(font_xref, width_table, def_width)
        new_fontsize += len(new_buffer)
    return old_fontsize - new_fontsize