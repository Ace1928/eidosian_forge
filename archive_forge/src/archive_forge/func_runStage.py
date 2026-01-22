import time
import pyqtgraph as pg
def runStage(i):
    """Waste time for 2 seconds while incrementing a progress bar.
    """
    with pg.ProgressDialog('Running stage %s..' % i, maximum=100, nested=True) as dlg:
        for j in range(100):
            time.sleep(0.02)
            dlg += 1
            if dlg.wasCanceled():
                print('Canceled stage %s' % i)
                break