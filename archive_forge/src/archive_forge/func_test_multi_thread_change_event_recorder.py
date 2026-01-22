import os
import shutil
import tempfile
import threading
import unittest
from traits.api import HasTraits, on_trait_change, Bool, Float, List
from traits import trait_notifiers
from traits.util.event_tracer import (
def test_multi_thread_change_event_recorder(self):
    test_object = TestObject()
    container = MultiThreadRecordContainer()
    recorder = MultiThreadChangeEventRecorder(container=container)
    trait_notifiers.set_change_event_tracers(pre_tracer=recorder.pre_tracer, post_tracer=recorder.post_tracer)
    try:
        test_object.number = 5.0
        thread = threading.Thread(target=test_object.add_to_number, args=(5,))
        thread.start()
        thread.join()
    finally:
        trait_notifiers.clear_change_event_tracers()
    self.assertEqual(len(container._record_containers), 2)
    container.save_to_directory(self.directory)
    for name in container._record_containers:
        filename = os.path.join(self.directory, '{0}.trace'.format(name))
        with open(filename, 'r', encoding='utf-8') as handle:
            lines = handle.readlines()
        self.assertEqual(len(lines), 4)
        if 'MainThread.trace' in filename:
            self.assertTrue("-> 'number' changed from 2.0 to 5.0 in 'TestObject'\n" in lines[0])
        else:
            self.assertTrue("-> 'number' changed from 5.0 to 10.0 in 'TestObject'\n" in lines[0])
        self.assertTrue('CALLING' in lines[1])
        self.assertTrue('EXIT' in lines[2])