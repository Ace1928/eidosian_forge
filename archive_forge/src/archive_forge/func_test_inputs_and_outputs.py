from unittest import TestCase
from .._methodical import MethodicalMachine
def test_inputs_and_outputs(self):
    traces = []

    def tracer(old_state, input, new_state):
        traces.append((old_state, input, new_state, None))

        def trace_outputs(output):
            traces.append((old_state, input, new_state, output))
        return trace_outputs
    s = SampleObject()
    s.setTrace(tracer)
    s.go1()
    self.assertEqual(traces, [('begin', 'go1', 'middle', None), ('begin', 'go1', 'middle', 'out')])
    s.go2()
    self.assertEqual(traces, [('begin', 'go1', 'middle', None), ('begin', 'go1', 'middle', 'out'), ('middle', 'go2', 'end', None), ('middle', 'go2', 'end', 'out')])
    s.setTrace(None)
    s.back()
    self.assertEqual(traces, [('begin', 'go1', 'middle', None), ('begin', 'go1', 'middle', 'out'), ('middle', 'go2', 'end', None), ('middle', 'go2', 'end', 'out')])
    s.go2()
    self.assertEqual(traces, [('begin', 'go1', 'middle', None), ('begin', 'go1', 'middle', 'out'), ('middle', 'go2', 'end', None), ('middle', 'go2', 'end', 'out')])