import numpy
import cv2
import time
import pygame
def list_cameras_darwin():
    import subprocess
    from xml.etree import ElementTree
    flout, _ = subprocess.Popen('system_profiler -xml SPCameraDataType', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    last_text = None
    cameras = []
    for node in ElementTree.fromstring(flout).iterfind('./array/dict/array/dict/*'):
        if last_text == '_name':
            cameras.append(node.text)
        last_text = node.text
    return cameras