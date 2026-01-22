import base64
import hashlib
import io
import struct
import time
import zipfile
from vtk.vtkCommonCore import vtkTypeInt32Array, vtkTypeUInt32Array
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkFiltersGeometry import (
from vtk.vtkRenderingCore import vtkColorTransferFunction
from .enums import TextPosition
def initializeSerializers():
    registerInstanceSerializer('vtkCornerAnnotation', annotationSerializer)
    registerInstanceSerializer('vtkImageSlice', genericProp3DSerializer)
    registerInstanceSerializer('vtkVolume', genericProp3DSerializer)
    registerInstanceSerializer('vtkOpenGLActor', genericActorSerializer)
    registerInstanceSerializer('vtkFollower', genericActorSerializer)
    registerInstanceSerializer('vtkPVLODActor', genericActorSerializer)
    registerInstanceSerializer('vtkOpenGLPolyDataMapper', genericPolyDataMapperSerializer)
    registerInstanceSerializer('vtkCompositePolyDataMapper2', genericPolyDataMapperSerializer)
    registerInstanceSerializer('vtkDataSetMapper', genericPolyDataMapperSerializer)
    registerInstanceSerializer('vtkFixedPointVolumeRayCastMapper', genericVolumeMapperSerializer)
    registerInstanceSerializer('vtkSmartVolumeMapper', genericVolumeMapperSerializer)
    registerInstanceSerializer('vtkOpenGLImageSliceMapper', imageSliceMapperSerializer)
    registerInstanceSerializer('vtkOpenGLGlyph3DMapper', glyph3DMapperSerializer)
    registerInstanceSerializer('vtkLookupTable', lookupTableSerializer2)
    registerInstanceSerializer('vtkPVDiscretizableColorTransferFunction', colorTransferFunctionSerializer)
    registerInstanceSerializer('vtkColorTransferFunction', colorTransferFunctionSerializer)
    registerInstanceSerializer('vtkPiecewiseFunction', piecewiseFunctionSerializer)
    registerInstanceSerializer('vtkOpenGLTexture', textureSerializer)
    registerInstanceSerializer('vtkOpenGLProperty', propertySerializer)
    registerInstanceSerializer('vtkVolumeProperty', volumePropertySerializer)
    registerInstanceSerializer('vtkImageProperty', imagePropertySerializer)
    registerInstanceSerializer('vtkPolyData', polydataSerializer)
    registerInstanceSerializer('vtkImageData', imageDataSerializer)
    registerInstanceSerializer('vtkStructuredGrid', mergeToPolydataSerializer)
    registerInstanceSerializer('vtkUnstructuredGrid', mergeToPolydataSerializer)
    registerInstanceSerializer('vtkMultiBlockDataSet', mergeToPolydataSerializer)
    registerInstanceSerializer('vtkCocoaRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkXOpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkWin32OpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkEGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkOpenVRRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkGenericOpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkOSOpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkOpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkIOSRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkExternalOpenGLRenderWindow', renderWindowSerializer)
    registerInstanceSerializer('vtkOpenGLRenderer', rendererSerializer)
    registerInstanceSerializer('vtkOpenGLCamera', cameraSerializer)