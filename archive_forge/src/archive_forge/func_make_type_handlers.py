import json
import re
import struct
import zipfile
import vtk
from .synchronizable_serializer import arrayTypesMapping
def make_type_handlers():
    aliases = {'vtkMapper': ['vtkOpenGLPolyDataMapper', 'vtkCompositePolyDataMapper2', 'vtkDataSetMapper'], 'vtkProperty': ['vtkOpenGLProperty'], 'vtkRenderer': ['vtkOpenGLRenderer'], 'vtkCamera': ['vtkOpenGLCamera'], 'vtkColorTransferFunction': ['vtkPVDiscretizableColorTransferFunction'], 'vtkActor': ['vtkOpenGLActor', 'vtkPVLODActor'], 'vtkLight': ['vtkOpenGLLight', 'vtkPVLight'], 'vtkTexture': ['vtkOpenGLTexture'], 'vtkVolumeMapper': ['vtkFixedPointVolumeRayCastMapper', 'vtkSmartVolumeMapper'], 'vtkGlyph3DMapper': ['vtkOpenGLGlyph3DMapper']}
    type_handlers = {'vtkRenderer': generic_builder, 'vtkLookupTable': generic_builder, 'vtkLight': None, 'vtkCamera': generic_builder, 'vtkPolyData': poly_data_builder, 'vtkImageData': generic_builder, 'vtkMapper': generic_builder, 'vtkGlyph3DMapper': generic_builder, 'vtkProperty': generic_builder, 'vtkActor': generic_builder, 'vtkFollower': generic_builder, 'vtkColorTransferFunction': color_fun_builder, 'vtkPiecewiseFunction': piecewise_fun_builder, 'vtkTexture': generic_builder, 'vtkVolumeMapper': volume_mapper_builder, 'vtkVolume': generic_builder, 'vtkVolumeProperty': generic_builder}
    for k, alias_list in aliases.items():
        for alias in alias_list:
            type_handlers.update({alias: type_handlers[k]})
    return type_handlers