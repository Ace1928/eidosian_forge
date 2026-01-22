import py
import os, sys
def dokill(pid):
    py.process.cmdexec('taskkill /F /PID %d' % (pid,))