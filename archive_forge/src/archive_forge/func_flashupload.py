import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
@cherrypy.expose
def flashupload(self, Filedata, Upload, Filename):
    return 'Upload: %s, Filename: %s, Filedata: %r' % (Upload, Filename, Filedata.file.read())