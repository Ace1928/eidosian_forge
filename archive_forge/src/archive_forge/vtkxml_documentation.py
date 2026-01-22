import numpy as np

    from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray
    iia = vtkImageImportFromArray()
    #iia.SetArray(Numeric_asarray(data.swapaxes(0,2).flatten()))
    iia.SetArray(Numeric_asarray(data))
    ida = iia.GetOutput()
    ipd = ida.GetPointData()
    ipd.SetName('scalars')
    spd.SetScalars(ipd.GetScalars())
    