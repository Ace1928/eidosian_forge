from typing import Dict
def convert_font_metrics(path: str) -> None:
    """Convert an AFM file to a mapping of font metrics.

    See below for the output.
    """
    fonts = {}
    with open(path, 'r') as fileinput:
        for line in fileinput.readlines():
            f = line.strip().split(' ')
            if not f:
                continue
            k = f[0]
            if k == 'FontName':
                fontname = f[1]
                props = {'FontName': fontname, 'Flags': 0}
                chars: Dict[int, int] = {}
                fonts[fontname] = (props, chars)
            elif k == 'C':
                cid = int(f[1])
                if 0 <= cid and cid <= 255:
                    width = int(f[4])
                    chars[cid] = width
            elif k in ('CapHeight', 'XHeight', 'ItalicAngle', 'Ascender', 'Descender'):
                k = {'Ascender': 'Ascent', 'Descender': 'Descent'}.get(k, k)
                props[k] = float(f[1])
            elif k in ('FontName', 'FamilyName', 'Weight'):
                k = {'FamilyName': 'FontFamily', 'Weight': 'FontWeight'}.get(k, k)
                props[k] = f[1]
            elif k == 'IsFixedPitch':
                if f[1].lower() == 'true':
                    props['Flags'] = 64
            elif k == 'FontBBox':
                props[k] = tuple(map(float, f[1:5]))
        print('# -*- python -*-')
        print('FONT_METRICS = {')
        for fontname, (props, chars) in fonts.items():
            print(' {!r}: {!r},'.format(fontname, (props, chars)))
        print('}')