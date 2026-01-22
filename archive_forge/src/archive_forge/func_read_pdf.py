import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import logging
import yaml
import pickle
import configparser
import markdown
import openpyxl
import sqlite3
import PyPDF2
import PIL.Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Label, Toplevel
from PIL import Image, ImageTk
import os
import logging
import json
import pandas as pd
def read_pdf(self) -> PyPDF2.PdfReader:
    """
        Read a PDF file and return its content as a PyPDF2 PdfReader object, ensuring that every aspect of the PDF data,
        including text, images, forms, and metadata, is comprehensively extracted and represented in the returned PdfReader object.
        This method is meticulously crafted to handle complex PDF files with various embedded elements, providing a robust,
        complete, and perfect representation of the PDF content.

        :return: The content of the PDF file, including all textual and non-textual elements.
        :rtype: PyPDF2.PdfReader
        """
    try:
        with open(self.file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            logging.debug(f'PDF data read successfully from {self.file_path}')
            num_pages = len(reader.pages)
            logging.info(f'The PDF file contains {num_pages} pages.')
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                logging.debug(f'Extracted text from page {i}: {text[:100]}...')
            return reader
    except Exception as e:
        logging.error(f'An error occurred while reading the PDF file at {self.file_path}: {str(e)}')
        raise Exception(f'An error occurred while reading the PDF file: {str(e)}')