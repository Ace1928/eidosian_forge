from reportlab.rl_config import register_reset
def createBarcodeDrawing(codeName, **options):
    """This creates and returns a drawing with a barcode.
    """
    from reportlab.graphics.shapes import Drawing
    codes = getCodes()
    bcc = codes[codeName]
    width = options.pop('width', None)
    height = options.pop('height', None)
    isoScale = options.pop('isoScale', 0)
    kw = {}
    for k, v in options.items():
        if k.startswith('_') or k in bcc._attrMap:
            kw[k] = v
    bc = bcc(**kw)
    if hasattr(bc, 'validate'):
        bc.validate()
        if not bc.valid:
            raise ValueError("Illegal barcode with value '%s' in code '%s'" % (options.get('value', None), codeName))
    x1, y1, x2, y2 = bc.getBounds()
    w = float(x2 - x1)
    h = float(y2 - y1)
    sx = width not in ('auto', None)
    sy = height not in ('auto', None)
    if sx or sy:
        sx = sx and width / w or 1.0
        sy = sy and height / h or 1.0
        if isoScale:
            if sx < 1.0 and sy < 1.0:
                sx = sy = max(sx, sy)
            else:
                sx = sy = min(sx, sy)
        w *= sx
        h *= sy
    else:
        sx = sy = 1
    d = Drawing(width=w, height=h, transform=[sx, 0, 0, sy, -sx * x1, -sy * y1])
    d.add(bc, '_bc')
    return d