import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def commit_database_changes(conn):
    try:
        conn.commit()
    except sqlite3.Error as e:
        print(f'Error committing changes: {e}')