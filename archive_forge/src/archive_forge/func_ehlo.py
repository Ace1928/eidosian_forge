import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def ehlo(self):
    return (200, 'Ok')