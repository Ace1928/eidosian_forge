import argparse
import ast
import re
import sys
def get_print_tensor_argparser(description):
    """Get an ArgumentParser for a command that prints tensor values.

  Examples of such commands include print_tensor and print_feed.

  Args:
    description: Description of the ArgumentParser.

  Returns:
    An instance of argparse.ArgumentParser.
  """
    ap = argparse.ArgumentParser(description=description, usage=argparse.SUPPRESS)
    ap.add_argument('tensor_name', type=str, help='Name of the tensor, followed by any slicing indices, e.g., hidden1/Wx_plus_b/MatMul:0, hidden1/Wx_plus_b/MatMul:0[1, :]')
    ap.add_argument('-n', '--number', dest='number', type=int, default=-1, help='0-based dump number for the specified tensor. Required for tensor with multiple dumps.')
    ap.add_argument('-r', '--ranges', dest='ranges', type=str, default='', help='Numerical ranges to highlight tensor elements in. Examples: -r 0,1e-8, -r [-0.1,0.1], -r "[[-inf, -0.1], [0.1, inf]]"')
    ap.add_argument('-a', '--all', dest='print_all', action='store_true', help='Print the tensor in its entirety, i.e., do not use ellipses.')
    ap.add_argument('-s', '--numeric_summary', action='store_true', help='Include summary for non-empty tensors of numeric (int*, float*, complex*) and Boolean types.')
    ap.add_argument('-w', '--write_path', type=str, default='', help='Path of the numpy file to write the tensor data to, using numpy.save().')
    return ap