import re, sys, os, tempfile, json

    To avoid memory leaks and random PARI crashes, runs CyPHC
    in a separate subprocess.
    