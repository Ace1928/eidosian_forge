import enum
Expected type of the results from a engine job call.

    Since programs have an embedded Any field, different types
    of data can be passed into a program/job for execution.
    This enum tracks the type of data that was passed in during
    the initial call so that the results can be handled appropriately.

    Program: A single circuit with a single TrialResult.
    Batch: A list of circuits with a list of TrialResults in a BatchResult.
    Calibration: List of CalibrationLayers returning a list of CalibrationResult
    