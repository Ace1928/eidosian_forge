from .gui import Tk_, ttk, SimpleDialog
def buttonbox(self):
    box = ttk.Frame(self)
    w = ttk.Button(box, text='OK', width=10, command=self.ok, default=Tk_.ACTIVE)
    w.pack(side=Tk_.LEFT, padx=5, pady=5)
    self.bind('<Return>', self.ok)
    self.bind('<Escape>', self.ok)
    box.grid(row=1, columnspan=2)