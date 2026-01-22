from collections import OrderedDict
import importlib

This module is a collection of classes that provide a
friendlier interface to MPI (through mpi4py). They help
allocate local tasks/data from global tasks/data and gather
global data (from all processors).

Although general, this module was only implemented to 
work with the convergence evaluation framework. More work
is needed to make this appropriate for general use.
