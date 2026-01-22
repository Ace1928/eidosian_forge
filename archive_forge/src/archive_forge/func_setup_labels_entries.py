import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip
def setup_labels_entries(self):
    """
        Setup labels and entry widgets for user input on coordinates and areas.
        """
    self.label_mouse_position = Label(self.root, textvariable=self.mouse_position)
    self.label_chat1 = Label(self.root, text='Chat Window 1 Coordinates (x, y):')
    self.entry_chat1_x = Entry(self.root)
    self.entry_chat1_y = Entry(self.root)
    self.label_chat2 = Label(self.root, text='Chat Window 2 Coordinates (x, y):')
    self.entry_chat2_x = Entry(self.root)
    self.entry_chat2_y = Entry(self.root)
    self.label_area = Label(self.root, text='Capture Area (width, height):')
    self.entry_area_width = Entry(self.root)
    self.entry_area_height = Entry(self.root)
    self.label_mouse_position.grid(row=0, column=0, columnspan=3)
    self.label_chat1.grid(row=1, column=0)
    self.entry_chat1_x.grid(row=1, column=1)
    self.entry_chat1_y.grid(row=1, column=2)
    self.label_chat2.grid(row=2, column=0)
    self.entry_chat2_x.grid(row=2, column=1)
    self.entry_chat2_y.grid(row=2, column=2)
    self.label_area.grid(row=3, column=0)
    self.entry_area_width.grid(row=3, column=1)
    self.entry_area_height.grid(row=3, column=2)