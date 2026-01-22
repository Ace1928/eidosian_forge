from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
def doesnt_change_before_cursor(completion):
    end = completion.text[:-completion.start_position]
    return document.text_before_cursor.endswith(end)