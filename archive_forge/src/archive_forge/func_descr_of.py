import sys
import cv2 as cv
@register('cv2.gapi')
def descr_of(*args):
    return [*args]