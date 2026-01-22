import sys, io
Push 'line' (a string) onto an internal buffer that will be
           checked by future 'readline()' calls.  Handy for implementing
           a parser with line-at-a-time lookahead.