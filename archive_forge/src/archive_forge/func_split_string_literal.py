from __future__ import absolute_import
import re
import sys
def split_string_literal(s, limit=2000):
    if len(s) < limit:
        return s
    else:
        start = 0
        chunks = []
        while start < len(s):
            end = start + limit
            if len(s) > end - 4 and '\\' in s[end - 4:end]:
                end -= 4 - s[end - 4:end].find('\\')
                while s[end - 1] == '\\':
                    end -= 1
                    if end == start:
                        end = start + limit - limit % 2 - 4
                        break
            chunks.append(s[start:end])
            start = end
        return '""'.join(chunks)