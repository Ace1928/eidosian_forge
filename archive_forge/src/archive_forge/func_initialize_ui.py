import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip
def initialize_ui(self):
    """
        Initialize the user interface for setting up chat window coordinates and interaction areas.
        """
    self.root.title('Chat Automation Setup')
    self.setup_labels_entries()
    self.setup_buttons()
    self.display_mouse_position()