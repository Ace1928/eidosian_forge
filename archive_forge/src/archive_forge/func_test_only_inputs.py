from unittest import TestCase
from .._methodical import MethodicalMachine
def test_only_inputs(self):
    traces = []

    def tracer(old_state, input, new_state):
        traces.append((old_state, input, new_state))
        return None
    s = SampleObject()
    s.setTrace(tracer)
    s.go1()
    self.assertEqual(traces, [('begin', 'go1', 'middle')])
    s.go2()
    self.assertEqual(traces, [('begin', 'go1', 'middle'), ('middle', 'go2', 'end')])
    s.setTrace(None)
    s.back()
    self.assertEqual(traces, [('begin', 'go1', 'middle'), ('middle', 'go2', 'end')])
    s.go2()
    self.assertEqual(traces, [('begin', 'go1', 'middle'), ('middle', 'go2', 'end')])