import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def save_as_txt(messages, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for msg in messages:
            file.write(f'{msg}\n')