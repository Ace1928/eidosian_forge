import sys
import select
def inputhook_mac(app=None):
    if self.pyplot_imported:
        pyplot = sys.modules['matplotlib.pyplot']
        try:
            pyplot.pause(0.01)
        except:
            pass
    elif 'matplotlib.pyplot' in sys.modules:
        self.pyplot_imported = True