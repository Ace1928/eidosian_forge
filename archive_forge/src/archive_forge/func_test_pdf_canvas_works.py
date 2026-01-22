import unittest
import pidtest
from rdkit.sping.PS import pidPS
def test_pdf_canvas_works(self):
    from rdkit.Chem.Draw.spingCanvas import Canvas
    canvas = Canvas((200, 400), 'test.pdf', imageType='pdf')
    canvas.canvas.showPage()