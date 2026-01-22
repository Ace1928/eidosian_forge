from pyparsing import *
import urllib.request, urllib.parse, urllib.error
A special subclass of Token that does *close* matches. For each
       close match of the given string, a tuple is returned giving the
       found close match, and a list of mismatch positions.