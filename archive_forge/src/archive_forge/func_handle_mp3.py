from twisted import copyright
from twisted.web import http
def handle_mp3(self):
    if len(self.databuffer) > self.metaint:
        self.gotMP3Data(self.databuffer[:self.metaint])
        self.databuffer = self.databuffer[self.metaint:]
        self.metamode = 'length'
    else:
        return 1