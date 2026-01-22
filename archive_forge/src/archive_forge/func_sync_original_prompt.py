from pexpect import ExceptionPexpect, TIMEOUT, EOF, spawn
import time
import os
import sys
import re
def sync_original_prompt(self, sync_multiplier=1.0):
    """This attempts to find the prompt. Basically, press enter and record
        the response; press enter again and record the response; if the two
        responses are similar then assume we are at the original prompt.
        This can be a slow function. Worst case with the default sync_multiplier
        can take 12 seconds. Low latency connections are more likely to fail
        with a low sync_multiplier. Best case sync time gets worse with a
        high sync multiplier (500 ms with default). """
    self.sendline()
    time.sleep(0.1)
    try:
        self.try_read_prompt(sync_multiplier)
    except TIMEOUT:
        pass
    self.sendline()
    x = self.try_read_prompt(sync_multiplier)
    self.sendline()
    a = self.try_read_prompt(sync_multiplier)
    self.sendline()
    b = self.try_read_prompt(sync_multiplier)
    ld = self.levenshtein_distance(a, b)
    len_a = len(a)
    if len_a == 0:
        return False
    if float(ld) / len_a < 0.4:
        return True
    return False