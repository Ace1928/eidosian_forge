import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
Setup connection to remote server on "host:port".
                (default: localhost:standard IMAP4 SSL port).
            This connection will be used by the routines:
                read, readline, send, shutdown.
            