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
def show_dataframe(self, df):
    self.data_tree.delete(*self.data_tree.get_children())
    self.data_tree['columns'] = list(df.columns)
    for col in df.columns:
        self.data_tree.heading(col, text=col)
    for row in df.to_records(index=False):
        self.data_tree.insert('', 'end', values=row)