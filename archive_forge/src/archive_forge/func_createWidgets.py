from sping import colors
from sping.TK import TKCanvas, TKCanvasPIL
from Tkinter import *
def createWidgets(self):
    self.question = Label(self, text='Can Find The BLUE Square??????')
    self.question.pack()
    self.QUIT = Button(self, text='QUIT', height=3, command=self.quit)
    self.QUIT.pack(side=BOTTOM)
    Button(self, text='Save to jpeg file', command=self.saveToJpeg).pack(side=BOTTOM)
    spacer = Frame(self, height='0.25i')
    spacer.pack(side=BOTTOM)
    self.tkpil = TKCanvasPIL(name='SpingTKCanvas', size=(600, 600), master=self, scrollingViewPortSize=(400, 400))
    self.draw = self.tkpil.getTKCanvas()
    self.draw.scrollX = Scrollbar(self, orient=HORIZONTAL)
    self.draw.scrollY = Scrollbar(self, orient=VERTICAL)
    self.draw['xscrollcommand'] = self.draw.scrollX.set
    self.draw['yscrollcommand'] = self.draw.scrollY.set
    self.draw.scrollX['command'] = self.draw.xview
    self.draw.scrollY['command'] = self.draw.yview
    self.tkpil.drawRect(10, 10, 100, 100, edgeColor=colors.blue, fillColor=colors.green)
    self.tkpil.drawRect(400, 400, 500, 500, edgeColor=colors.blue, fillColor=colors.lightblue)
    self.tkpil.drawRect(30, 400, 130, 500, edgeColor=colors.blue, fillColor=colors.yellow)
    self.tkpil.flush()
    self.draw.scrollX.pack(side=BOTTOM, fill=X)
    self.draw.scrollY.pack(side=RIGHT, fill=Y)
    self.draw.pack(side=LEFT)