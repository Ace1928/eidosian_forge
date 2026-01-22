import logging
import sys
from io import StringIO
from typing import Any, BinaryIO, Container, Iterator, Optional, cast
from .converter import (
from .image import ImageWriter
from .layout import LAParams, LTPage
from .pdfdevice import PDFDevice, TagExtractor
from .pdfinterp import PDFResourceManager, PDFPageInterpreter
from .pdfpage import PDFPage
from .utils import open_filename, FileOrName, AnyIO
def extract_text_to_fp(inf: BinaryIO, outfp: AnyIO, output_type: str='text', codec: str='utf-8', laparams: Optional[LAParams]=None, maxpages: int=0, page_numbers: Optional[Container[int]]=None, password: str='', scale: float=1.0, rotation: int=0, layoutmode: str='normal', output_dir: Optional[str]=None, strip_control: bool=False, debug: bool=False, disable_caching: bool=False, **kwargs: Any) -> None:
    """Parses text from inf-file and writes to outfp file-like object.

    Takes loads of optional arguments but the defaults are somewhat sane.
    Beware laparams: Including an empty LAParams is not the same as passing
    None!

    :param inf: a file-like object to read PDF structure from, such as a
        file handler (using the builtin `open()` function) or a `BytesIO`.
    :param outfp: a file-like object to write the text to.
    :param output_type: May be 'text', 'xml', 'html', 'hocr', 'tag'.
        Only 'text' works properly.
    :param codec: Text decoding codec
    :param laparams: An LAParams object from pdfminer.layout. Default is None
        but may not layout correctly.
    :param maxpages: How many pages to stop parsing after
    :param page_numbers: zero-indexed page numbers to operate on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param scale: Scale factor
    :param rotation: Rotation factor
    :param layoutmode: Default is 'normal', see
        pdfminer.converter.HTMLConverter
    :param output_dir: If given, creates an ImageWriter for extracted images.
    :param strip_control: Does what it says on the tin
    :param debug: Output more logging data
    :param disable_caching: Does what it says on the tin
    :param other:
    :return: nothing, acting as it does on two streams. Use StringIO to get
        strings.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)
    rsrcmgr = PDFResourceManager(caching=not disable_caching)
    device: Optional[PDFDevice] = None
    if output_type != 'text' and outfp == sys.stdout:
        outfp = sys.stdout.buffer
    if output_type == 'text':
        device = TextConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, imagewriter=imagewriter)
    elif output_type == 'xml':
        device = XMLConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, imagewriter=imagewriter, stripcontrol=strip_control)
    elif output_type == 'html':
        device = HTMLConverter(rsrcmgr, outfp, codec=codec, scale=scale, layoutmode=layoutmode, laparams=laparams, imagewriter=imagewriter)
    elif output_type == 'hocr':
        device = HOCRConverter(rsrcmgr, outfp, codec=codec, laparams=laparams, stripcontrol=strip_control)
    elif output_type == 'tag':
        device = TagExtractor(rsrcmgr, cast(BinaryIO, outfp), codec=codec)
    else:
        msg = f'Output type can be text, html, xml or tag but is {output_type}'
        raise ValueError(msg)
    assert device is not None
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(inf, page_numbers, maxpages=maxpages, password=password, caching=not disable_caching):
        page.rotate = (page.rotate + rotation) % 360
        interpreter.process_page(page)
    device.close()