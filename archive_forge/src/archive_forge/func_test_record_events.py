import os
import shutil
import tempfile
import threading
import unittest
from traits.api import HasTraits, on_trait_change, Bool, Float, List
from traits import trait_notifiers
from traits.util.event_tracer import (
def test_record_events(self):
    test_object = TestObject()
    with record_events() as container:
        test_object.number = 5.0
        thread = threading.Thread(target=test_object.add_to_number, args=(3,))
        thread.start()
        thread.join()
    container.save_to_directory(self.directory)
    for name in container._record_containers:
        filename = os.path.join(self.directory, '{0}.trace'.format(name))
        with open(filename, 'r', encoding='utf-8') as handle:
            lines = handle.readlines()
        self.assertEqual(len(lines), 4)
        if 'MainThread.trace' in filename:
            self.assertTrue("-> 'number' changed from 2.0 to 5.0 in 'TestObject'\n" in lines[0])
        else:
            self.assertTrue("-> 'number' changed from 5.0 to 8.0 in 'TestObject'\n" in lines[0])
        self.assertTrue('CALLING' in lines[1])
        self.assertTrue('EXIT' in lines[2])