import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def rTreeFromChromRangeArray(self, blockSize, items, endFileOffset):
    itemCount = len(items)
    if itemCount == 0:
        return
    children = []
    nextOffset = items[0].offset
    oneSize = 0
    i = 0
    while i < itemCount:
        child = _RTreeNode()
        children.append(child)
        startItem = items[i]
        child.startChromId = child.endChromId = startItem.chromId
        child.startBase = startItem.start
        child.endBase = startItem.end
        child.startFileOffset = nextOffset
        oneSize = 1
        endItem = startItem
        for j in range(i + 1, itemCount):
            endItem = items[j]
            nextOffset = endItem.offset
            if nextOffset != child.startFileOffset:
                break
            oneSize += 1
        else:
            nextOffset = endFileOffset
        child.endFileOffset = nextOffset
        for item in items[i + 1:i + oneSize]:
            if item.chromId < child.startChromId:
                child.startChromId = item.chromId
                child.startBase = item.start
            elif item.chromId == child.startChromId and item.start < child.startBase:
                child.startBase = item.start
            if item.chromId > child.endChromId:
                child.endChromId = item.chromId
                child.endBase = item.end
            elif item.chromId == child.endChromId and item.end > child.endBase:
                child.endBase = item.end
        i += oneSize
    levelCount = 1
    while True:
        parents = []
        slotsUsed = blockSize
        for child in children:
            if slotsUsed >= blockSize:
                slotsUsed = 1
                parent = _RTreeNode()
                parent.parent = child.parent
                parent.startChromId = child.startChromId
                parent.startBase = child.startBase
                parent.endChromId = child.endChromId
                parent.endBase = child.endBase
                parent.startFileOffset = child.startFileOffset
                parent.endFileOffset = child.endFileOffset
                parents.append(parent)
            else:
                slotsUsed += 1
                if child.startChromId < parent.startChromId:
                    parent.startChromId = child.startChromId
                    parent.startBase = child.startBase
                elif child.startChromId == parent.startChromId and child.startBase < parent.startBase:
                    parent.startBase = child.startBase
                if child.endChromId > parent.endChromId:
                    parent.endChromId = child.endChromId
                    parent.endBase = child.endBase
                elif child.endChromId == parent.endChromId and child.endBase > parent.endBase:
                    parent.endBase = child.endBase
            parent.children.append(child)
            child.parent = parent
        levelCount += 1
        if len(parents) == 1:
            break
        children = parents
    return (parent, levelCount)