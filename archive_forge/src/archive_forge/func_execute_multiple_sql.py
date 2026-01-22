import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def execute_multiple_sql(conn, script):
    try:
        with conn.cursor() as cursor:
            cursor.executescript(script)
            conn.commit()
    except sqlite3.Error as e:
        print(f'Error executing script: {e}')