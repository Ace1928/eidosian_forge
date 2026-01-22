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
Extract and yield LTPage objects

    :param pdf_file: Either a file path or a file-like object for the PDF file
        to be worked on.
    :param password: For encrypted PDFs, the password to decrypt.
    :param page_numbers: List of zero-indexed page numbers to extract.
    :param maxpages: The maximum number of pages to parse
    :param caching: If resources should be cached
    :param laparams: An LAParams object from pdfminer.layout. If None, uses
        some default settings that often work well.
    :return: LTPage objects
    