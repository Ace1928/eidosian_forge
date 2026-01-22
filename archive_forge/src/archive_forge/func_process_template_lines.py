import os
import sys
from setuptools import setup
def process_template_lines(template_lines):
    for version in ('38', '39'):
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        for line in template_lines:
            if version == '38':
                line = line.replace('get_bytecode_while_frame_eval(PyFrameObject * frame_obj, int exc)', 'get_bytecode_while_frame_eval_38(PyFrameObject * frame_obj, int exc)')
                line = line.replace('CALL_EvalFrameDefault', 'CALL_EvalFrameDefault_38(frame_obj, exc)')
            else:
                line = line.replace('get_bytecode_while_frame_eval(PyFrameObject * frame_obj, int exc)', 'get_bytecode_while_frame_eval_39(PyThreadState* tstate, PyFrameObject * frame_obj, int exc)')
                line = line.replace('CALL_EvalFrameDefault', 'CALL_EvalFrameDefault_39(tstate, frame_obj, exc)')
            yield line
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield ''
        yield ''