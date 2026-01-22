import os
import smtplib
from breezy import gpg, merge_directive, tests, workingtree
def run_bzr_fakemail(self, *args, **kwargs):
    sendmail_calls = []

    def sendmail(self, from_, to, message):
        sendmail_calls.append((self, from_, to, message))
    connect_calls = []

    def connect(self, host='localhost', port=0):
        connect_calls.append((self, host, port))
        return (220, 'Ok')

    def has_extn(self, extension):
        return False

    def ehlo(self):
        return (200, 'Ok')
    old_sendmail = smtplib.SMTP.sendmail
    smtplib.SMTP.sendmail = sendmail
    old_connect = smtplib.SMTP.connect
    smtplib.SMTP.connect = connect
    old_ehlo = smtplib.SMTP.ehlo
    smtplib.SMTP.ehlo = ehlo
    old_has_extn = smtplib.SMTP.has_extn
    smtplib.SMTP.has_extn = has_extn
    try:
        result = self.run_bzr(*args, **kwargs)
    finally:
        smtplib.SMTP.sendmail = old_sendmail
        smtplib.SMTP.connect = old_connect
        smtplib.SMTP.ehlo = old_ehlo
        smtplib.SMTP.has_extn = old_has_extn
    return result + (connect_calls, sendmail_calls)