import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def execute_parameterized_query(conn, query, params):
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    except sqlite3.Error as e:
        print(f'Error executing query: {e}')