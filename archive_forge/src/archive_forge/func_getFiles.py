import xml.dom.minidom, xml.sax.saxutils
import os, time, fcntl
from xdg.Exceptions import ParsingError
def getFiles(self, mimetypes=None, groups=None, limit=0):
    """Get a list of recently used files.
        
        The parameters can be used to filter by mime types, by group, or to
        limit the number of items returned. By default, the entire list is
        returned, except for items marked private.
        """
    tmp = []
    i = 0
    for item in self.RecentFiles:
        if groups:
            for group in groups:
                if group in item.Groups:
                    tmp.append(item)
                    i += 1
        elif mimetypes:
            for mimetype in mimetypes:
                if mimetype == item.MimeType:
                    tmp.append(item)
                    i += 1
        elif item.Private == False:
            tmp.append(item)
            i += 1
        if limit != 0 and i == limit:
            break
    return tmp