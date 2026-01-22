import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import requests
def optimize_database_interactions(conn):

    class DatabaseConnection:

        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self.conn.cursor()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()