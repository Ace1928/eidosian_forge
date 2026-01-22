import sys
import cv2 as cv
@register('cv2.gapi')
def compile_args(*args):
    return list(map(cv.GCompileArg, args))