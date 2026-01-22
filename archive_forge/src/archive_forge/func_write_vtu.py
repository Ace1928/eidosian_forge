import numpy as np
def write_vtu(filename, atoms, data=None):
    from vtk import VTK_MAJOR_VERSION, vtkUnstructuredGrid, vtkPoints, vtkXMLUnstructuredGridWriter
    from vtk.util.numpy_support import numpy_to_vtk
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a VTI file!')
        atoms = atoms[0]
    ugd = vtkUnstructuredGrid()
    p = vtkPoints()
    p.SetNumberOfPoints(len(atoms))
    p.SetDataTypeToDouble()
    for i, pos in enumerate(atoms.get_positions()):
        p.InsertPoint(i, *pos)
    ugd.SetPoints(p)
    numbers = numpy_to_vtk(atoms.get_atomic_numbers(), deep=1)
    ugd.GetPointData().AddArray(numbers)
    numbers.SetName('atomic numbers')
    tags = numpy_to_vtk(atoms.get_tags(), deep=1)
    ugd.GetPointData().AddArray(tags)
    tags.SetName('tags')
    from ase.data import covalent_radii
    radii = numpy_to_vtk(covalent_radii[atoms.numbers], deep=1)
    ugd.GetPointData().AddArray(radii)
    radii.SetName('radii')
    w = vtkXMLUnstructuredGridWriter()
    if fast:
        w.SetDataModeToAppend()
        w.EncodeAppendedDataOff()
    else:
        w.GetCompressor().SetCompressionLevel(0)
        w.SetDataModeToAscii()
    if isinstance(filename, str):
        w.SetFileName(filename)
    else:
        w.SetFileName(filename.name)
    if VTK_MAJOR_VERSION <= 5:
        w.SetInput(ugd)
    else:
        w.SetInputData(ugd)
    w.Write()