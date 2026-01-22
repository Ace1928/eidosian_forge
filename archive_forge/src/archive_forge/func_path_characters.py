from __future__ import absolute_import
def path_characters():
    """
        Returns a string containing valid URL path characters.
        """
    global _path_characters
    if _path_characters is None:

        def chars():
            for i in range(maxunicode):
                c = unichr(i)
                if c in '#/?':
                    continue
                try:
                    c.encode('utf-8')
                except UnicodeEncodeError:
                    continue
                yield c
        _path_characters = ''.join(chars())
    return _path_characters