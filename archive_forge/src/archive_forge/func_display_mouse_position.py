import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip
def display_mouse_position(self):
    """
        Continuously update and display the mouse position in real-time.
        """
    x, y = pyautogui.position()
    self.mouse_position.set(f'Mouse Position: ({x}, {y})')
    self.root.after(100, self.display_mouse_position)