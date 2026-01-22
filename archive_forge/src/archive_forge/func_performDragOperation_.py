from pyglet.window import key, mouse
from pyglet.libs.darwin.quartzkey import keymap, charmap
from pyglet.libs.darwin import cocoapy, NSPasteboardURLReadingFileURLsOnlyKey, NSLeftShiftKeyMask, NSRightShiftKeyMask, \
from .pyglet_textview import PygletTextView
@PygletView.method('B@')
def performDragOperation_(self, sender):
    pos = sender.draggingLocation()
    pasteboard = sender.draggingPasteboard()
    classes = NSArray.arrayWithObject_(NSURL)
    options = NSDictionary.dictionaryWithObject_forKey_(NSNumber.numberWithBool_(True), NSPasteboardURLReadingFileURLsOnlyKey)
    urls = pasteboard.readObjectsForClasses_options_(classes, options)
    url_count = urls.count()
    paths = []
    for i in range(url_count):
        fpath = urls.objectAtIndex_(i).fileSystemRepresentation()
        paths.append(fpath.decode())
    self._window.dispatch_event('on_file_drop', pos.x, pos.y, paths)