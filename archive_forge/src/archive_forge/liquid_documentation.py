from the loaded GIF image is copied to the colormap for the
import pygame as pg
import os
from math import sin
import time
from. i've snipped the sound and text stuff out.
BTW, here is the code from the BlitzBasic example this was derived
from. i've snipped the sound and text stuff out.
-----------------------------------------------------------------
; Brad@freedom2000.com

; Load a bmp pic (800x600) and slice it into 1600 squares
Graphics 640,480
SetBuffer BackBuffer()
bitmap$="f2kwarp.bmp"
pic=LoadAnimImage(bitmap$,20,15,0,1600)

; use SIN to move all 1600 squares around to give liquid effect
Repeat
f=0:w=w+10:If w=360 Then w=0
For y=0 To 599 Step 15
For x = 0 To 799 Step 20
f=f+1:If f=1600 Then f=0
DrawBlock pic,(x+(Sin(w+x)*40))/1.7+80,(y+(Sin(w+y)*40))/1.7+60,f
Next:Next:Flip:Cls
Until KeyDown(1)
