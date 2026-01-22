from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtCore():
    from PySide2.QtCore import Qt, QUrl, QDir
    from PySide2.QtCore import QRect, QSize, QPoint, QLocale, QByteArray
    from PySide2.QtCore import QMarginsF
    try:
        from PySide2.QtCore import Connection
    except ImportError:
        pass
    type_map.update({"' '": ' ', "'%'": '%', "'g'": 'g', '4294967295UL': 4294967295, 'CheckIndexOption.NoOption': Instance('PySide2.QtCore.QAbstractItemModel.CheckIndexOptions.NoOption'), 'false': False, 'list of QAbstractAnimation': typing.List[PySide2.QtCore.QAbstractAnimation], 'list of QAbstractState': typing.List[PySide2.QtCore.QAbstractState], 'long long': int, 'long': int, 'NULL': None, 'nullptr': None, 'PyByteArray': bytearray, 'PyBytes': bytes, 'PyCallable': typing.Callable, 'PyObject': object, 'PySequence': typing.Iterable, 'PySide2.QtCore.bool': bool, 'PySide2.QtCore.char': StringList, 'PySide2.QtCore.double': float, 'PySide2.QtCore.float': float, 'PySide2.QtCore.int': int, 'PySide2.QtCore.int32_t': int, 'PySide2.QtCore.int64_t': int, 'PySide2.QtCore.long long': int, 'PySide2.QtCore.long': int, 'PySide2.QtCore.QCborStreamReader.StringResult': typing.AnyStr, 'PySide2.QtCore.QChar': Char, 'PySide2.QtCore.qint16': int, 'PySide2.QtCore.qint32': int, 'PySide2.QtCore.qint64': int, 'PySide2.QtCore.qint8': int, 'PySide2.QtCore.qreal': float, 'PySide2.QtCore.QString': str, 'PySide2.QtCore.QStringList': StringList, 'PySide2.QtCore.quint16': int, 'PySide2.QtCore.quint32': int, 'PySide2.QtCore.quint64': int, 'PySide2.QtCore.quint8': int, 'PySide2.QtCore.QUrl.ComponentFormattingOptions': PySide2.QtCore.QUrl.ComponentFormattingOption, 'PySide2.QtCore.QVariant': Variant, 'PySide2.QtCore.short': int, 'PySide2.QtCore.signed char': Char, 'PySide2.QtCore.uchar': Char, 'PySide2.QtCore.uint32_t': int, 'PySide2.QtCore.unsigned char': Char, 'PySide2.QtCore.unsigned int': int, 'PySide2.QtCore.unsigned short': int, 'PyTypeObject': type, 'PyUnicode': typing.Text, 'Q_NULLPTR': None, 'QChar': Char, 'QDir.Filters(AllEntries | NoDotAndDotDot)': Instance('QDir.Filters(QDir.AllEntries | QDir.NoDotAndDotDot)'), 'QDir.SortFlags(Name | IgnoreCase)': Instance('QDir.SortFlags(QDir.Name | QDir.IgnoreCase)'), 'QGenericArgument((0))': ellipsis, 'QGenericArgument()': ellipsis, 'QGenericArgument(0)': ellipsis, 'QGenericArgument(NULL)': ellipsis, 'QGenericArgument(nullptr)': ellipsis, 'QGenericArgument(Q_NULLPTR)': ellipsis, 'QHash': typing.Dict, 'QJsonObject': typing.Dict[str, PySide2.QtCore.QJsonValue], 'QModelIndex()': Invalid('PySide2.QtCore.QModelIndex'), 'QModelIndexList': ModelIndexList, 'qptrdiff': int, 'QString': str, 'QString()': '', 'QStringList': StringList, 'QStringList()': [], 'QStringRef': str, 'Qt.HANDLE': int, 'quintptr': int, 'QUrl.FormattingOptions(PrettyDecoded)': Instance('QUrl.FormattingOptions(QUrl.PrettyDecoded)'), 'QVariant': Variant, 'QVariant()': Invalid(Variant), 'QVariant.Type': type, 'QVariantMap': typing.Dict[str, Variant]})
    try:
        type_map.update({'PySide2.QtCore.QMetaObject.Connection': PySide2.QtCore.Connection})
    except AttributeError:
        pass
    return locals()