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
def browse_file(self):
    file_path = filedialog.askopenfilename(title='Select a File', filetypes=[('All files', '*.*')])
    self.file_path.set(file_path)
    if file_path:
        self.status_label.config(text=f'Selected file: {os.path.basename(file_path)}')