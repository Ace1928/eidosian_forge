import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def fetch_all_data(cursor):
    return cursor.fetchall()