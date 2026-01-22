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
def read_ini(self) -> configparser.ConfigParser:
    """
        Read an INI file and return its content as a ConfigParser object, ensuring that every possible section,
        option, and value is meticulously extracted and represented in the returned ConfigParser object.
        This method is designed to handle complex INI files with multiple sections and nested configurations,
        providing a robust, complete, and perfect representation of the INI content.

        :return: The content of the INI file parsed into a ConfigParser, including all sections, options, and values.
        :rtype: configparser.ConfigParser
        """
    config = configparser.ConfigParser()
    try:
        config.read(self.file_path, encoding='utf-8')
        logging.debug(f'INI data read successfully from {self.file_path}')
        sections = config.sections()
        logging.info(f'INI file contains {len(sections)} sections: {sections}')
        for section in sections:
            options = config.options(section)
            logging.info(f"Section '{section}' contains {len(options)} options.")
            for option in options:
                value = config.get(section, option)
                logging.info(f"Option '{option}' in section '{section}' has value: {value}")
        return config
    except configparser.Error as e:
        logging.error(f'INI parsing error at {self.file_path}: {str(e)}')
        raise Exception(f'INI parsing error at {self.file_path}: {str(e)}')
    except Exception as e:
        logging.error(f'An error occurred while reading the INI file at {self.file_path}: {str(e)}')
        raise Exception(f'An error occurred while reading the INI file: {str(e)}')