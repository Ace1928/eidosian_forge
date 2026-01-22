import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
def getCplexStyleArrays(self, lp, senseDict=None, LpVarCategories=None, LpObjSenses=None, infBound=1e+20):
    """returns the arrays suitable to pass to a cdll Cplex
        or other solvers that are similar

        Copyright (c) Stuart Mitchell 2007
        """
    if senseDict is None:
        senseDict = {const.LpConstraintEQ: 'E', const.LpConstraintLE: 'L', const.LpConstraintGE: 'G'}
    if LpVarCategories is None:
        LpVarCategories = {const.LpContinuous: 'C', const.LpInteger: 'I'}
    if LpObjSenses is None:
        LpObjSenses = {const.LpMaximize: -1, const.LpMinimize: 1}
    import ctypes
    rangeCount = 0
    variables = list(lp.variables())
    numVars = len(variables)
    self.v2n = {variables[i]: i for i in range(numVars)}
    self.vname2n = {variables[i].name: i for i in range(numVars)}
    self.n2v = {i: variables[i] for i in range(numVars)}
    objSense = LpObjSenses[lp.sense]
    NumVarDoubleArray = ctypes.c_double * numVars
    objectCoeffs = NumVarDoubleArray()
    for v, val in lp.objective.items():
        objectCoeffs[self.v2n[v]] = val
    objectConst = ctypes.c_double(0.0)
    NumVarStrArray = ctypes.c_char_p * numVars
    colNames = NumVarStrArray()
    lowerBounds = NumVarDoubleArray()
    upperBounds = NumVarDoubleArray()
    initValues = NumVarDoubleArray()
    for v in lp.variables():
        colNames[self.v2n[v]] = to_string(v.name)
        initValues[self.v2n[v]] = 0.0
        if v.lowBound != None:
            lowerBounds[self.v2n[v]] = v.lowBound
        else:
            lowerBounds[self.v2n[v]] = -infBound
        if v.upBound != None:
            upperBounds[self.v2n[v]] = v.upBound
        else:
            upperBounds[self.v2n[v]] = infBound
    numRows = len(lp.constraints)
    NumRowDoubleArray = ctypes.c_double * numRows
    NumRowStrArray = ctypes.c_char_p * numRows
    NumRowCharArray = ctypes.c_char * numRows
    rhsValues = NumRowDoubleArray()
    rangeValues = NumRowDoubleArray()
    rowNames = NumRowStrArray()
    rowType = NumRowCharArray()
    self.c2n = {}
    self.n2c = {}
    i = 0
    for c in lp.constraints:
        rhsValues[i] = -lp.constraints[c].constant
        rangeValues[i] = 0.0
        rowNames[i] = to_string(c)
        rowType[i] = to_string(senseDict[lp.constraints[c].sense])
        self.c2n[c] = i
        self.n2c[i] = c
        i = i + 1
    coeffs = lp.coefficients()
    sparseMatrix = sparse.Matrix(list(range(numRows)), list(range(numVars)))
    for var, row, coeff in coeffs:
        sparseMatrix.add(self.c2n[row], self.vname2n[var], coeff)
    numels, mystartsBase, mylenBase, myindBase, myelemBase = sparseMatrix.col_based_arrays()
    elemBase = ctypesArrayFill(myelemBase, ctypes.c_double)
    indBase = ctypesArrayFill(myindBase, ctypes.c_int)
    startsBase = ctypesArrayFill(mystartsBase, ctypes.c_int)
    lenBase = ctypesArrayFill(mylenBase, ctypes.c_int)
    NumVarCharArray = ctypes.c_char * numVars
    columnType = NumVarCharArray()
    if lp.isMIP():
        for v in lp.variables():
            columnType[self.v2n[v]] = to_string(LpVarCategories[v.cat])
    self.addedVars = numVars
    self.addedRows = numRows
    return (numVars, numRows, numels, rangeCount, objSense, objectCoeffs, objectConst, rhsValues, rangeValues, rowType, startsBase, lenBase, indBase, elemBase, lowerBounds, upperBounds, initValues, colNames, rowNames, columnType, self.n2v, self.n2c)