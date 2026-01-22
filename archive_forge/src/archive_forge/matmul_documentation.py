import torch
from ... import cdiv, heuristics, jit
from ... import language as tl

    Generates the look-up table for incrementing pointers in the DSD/DDS matmul.
    Example (BLOCK=32, STEP=16)
    [[1, 0, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 1, 0, 0]]

    Then the offsets for A are
     [0 , 16, 32, 48] <- row 0
      \----/  \----/
      col=0   col=3
     [64, 80, 96, 112, 128, 144] <- row 1
      \----/   \----/  \------/
       col=1    col=2    col=3
     [160, 176, 192, 208]
    which leads to increments table
    [0, 16, 16, 16, || 64, 16, 16, 16, 16, 16, || 160, 16, 16, 16]

    Because B is dense, the offsets are
    [0, 16, 96, 112] <- row 0
    [32, 48, 64, 80]  <- row 1
    [0, 16, 64, 80]   <- row 2
    