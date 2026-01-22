import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def open_file_dialog(self):
    file_path = filedialog.askopenfilename(title='Select the sms.db file', filetypes=[('Database files', '*.db')])
    self.file_path_var.set(file_path)