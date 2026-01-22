import argparse
import logging
import sys
from typing import Any, Container, Iterable, List, Optional
import pdfminer.high_level
from pdfminer.layout import LAParams
from pdfminer.utils import AnyIO
A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.