import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
def on_window_flip(self, window):
    """Internal method to be called when the window have just displayed an
        image.
        When an image is showed, we decrement our framecount. If framecount is
        come to 0, we are taking the screenshot.

        The screenshot is done in a temporary place, and is compared to the
        original one -> test ok/ko.
        If no screenshot is available in the results directory, a new one will
        be created.
        """
    from kivy.base import EventLoop
    from tempfile import mkstemp
    from os.path import join, exists
    from os import unlink, close
    from shutil import move, copy
    self.framecount = getattr(self, 'framecount', 0) - 1
    if self.framecount > 0:
        return
    if not make_screenshots:
        EventLoop.stop()
        return
    reffn = None
    match = False
    try:
        fd, tmpfn = mkstemp(suffix='.png', prefix='kivyunit-')
        close(fd)
        unlink(tmpfn)
        self.test_counter += 1
        test_uid = '%s-%d.png' % ('_'.join(self.id().split('.')[-2:]), self.test_counter)
        log.info('Capturing screenshot for %s' % test_uid)
        tmpfn = window.screenshot(tmpfn)
        log.info('Capture saved at %s' % tmpfn)
        reffn = join(self.results_dir, test_uid)
        log.info('Compare with %s' % reffn)
        import inspect
        frame = inspect.getouterframes(inspect.currentframe())[6]
        sourcecodetab, line = inspect.getsourcelines(frame[0])
        line = frame[2] - line
        currentline = sourcecodetab[line]
        sourcecodetab[line] = '<span style="color: red;">%s</span>' % currentline
        sourcecode = ''.join(sourcecodetab)
        sourcecodetab[line] = '>>>>>>>>\n%s<<<<<<<<\n' % currentline
        sourcecodeask = ''.join(sourcecodetab)
        if not exists(reffn):
            log.info('No image reference, move %s as ref ?' % test_uid)
            if self.interactive_ask_ref(sourcecodeask, tmpfn, self.id()):
                move(tmpfn, reffn)
                tmpfn = reffn
                log.info('Image used as reference')
                match = True
            else:
                log.info('Image discarded')
        else:
            from kivy.core.image import Image as CoreImage
            s1 = CoreImage(tmpfn, keep_data=True)
            sd1 = s1.image._data[0].data
            s2 = CoreImage(reffn, keep_data=True)
            sd2 = s2.image._data[0].data
            if sd1 != sd2:
                log.critical('%s at render() #%d, images are different.' % (self.id(), self.test_counter))
                if self.interactive_ask_diff(sourcecodeask, tmpfn, reffn, self.id()):
                    log.critical('user ask to use it as ref.')
                    move(tmpfn, reffn)
                    tmpfn = reffn
                    match = True
                else:
                    self.test_failed = True
            else:
                match = True
        from os.path import join, dirname, exists, basename
        from os import mkdir
        build_dir = join(dirname(__file__), 'build')
        if not exists(build_dir):
            mkdir(build_dir)
        copy(reffn, join(build_dir, 'ref_%s' % basename(reffn)))
        if tmpfn != reffn:
            copy(tmpfn, join(build_dir, 'test_%s' % basename(reffn)))
        with open(join(build_dir, 'index.html'), 'at') as fd:
            color = '#ffdddd' if not match else '#ffffff'
            fd.write('<div style="background-color: %s">' % color)
            fd.write('<h2>%s #%d</h2>' % (self.id(), self.test_counter))
            fd.write('<table><tr><th>Reference</th><th>Test</th><th>Comment</th>')
            fd.write('<tr><td><img src="ref_%s"/></td>' % basename(reffn))
            if tmpfn != reffn:
                fd.write('<td><img src="test_%s"/></td>' % basename(reffn))
            else:
                fd.write('<td>First time, no comparison.</td>')
            fd.write('<td><pre>%s</pre></td>' % sourcecode)
            fd.write('</table></div>')
    finally:
        try:
            if reffn != tmpfn:
                unlink(tmpfn)
        except:
            pass
        EventLoop.stop()