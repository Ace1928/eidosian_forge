from reportlab.rl_config import register_reset
def createBarcodeImageInMemory(codeName, **options):
    """This creates and returns barcode as an image in memory.
    Takes same arguments as createBarcodeDrawing and also an
    optional format keyword which can be anything acceptable
    to Drawing.asString eg gif, pdf, tiff, py ......
    """
    format = options.pop('format', 'png')
    d = createBarcodeDrawing(codeName, **options)
    return d.asString(format)