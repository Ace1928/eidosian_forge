import logging
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
def make_alphabet_flow(i):
    f = lf.Flow('alphabet_%s' % i)
    start_value = 'A'
    end_value = 'Z'
    curr_value = start_value
    while ord(curr_value) <= ord(end_value):
        next_value = chr(ord(curr_value) + 1)
        if curr_value != end_value:
            f.add(EchoTask(name='echoer_%s' % curr_value, rebind={'value': curr_value}, provides=next_value))
        else:
            f.add(EchoTask(name='echoer_%s' % curr_value, rebind={'value': curr_value}))
        curr_value = next_value
    return f