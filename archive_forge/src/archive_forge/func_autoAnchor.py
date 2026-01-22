from ..Point import Point
def autoAnchor(self, pos, relative=True):
    """
        Set the position of this item relative to its parent by automatically 
        choosing appropriate anchor settings.
        
        If relative is True, one corner of the item will be anchored to 
        the appropriate location on the parent with no offset. The anchored
        corner will be whichever is closest to the parent's boundary.
        
        If relative is False, one corner of the item will be anchored to the same
        corner of the parent, with an absolute offset to achieve the correct
        position. 
        """
    pos = Point(pos)
    br = self.mapRectToParent(self.boundingRect()).translated(pos - self.pos())
    pbr = self.parentItem().boundingRect()
    anchorPos = [0, 0]
    parentPos = Point()
    itemPos = Point()
    if abs(br.left() - pbr.left()) < abs(br.right() - pbr.right()):
        anchorPos[0] = 0
        parentPos[0] = pbr.left()
        itemPos[0] = br.left()
    else:
        anchorPos[0] = 1
        parentPos[0] = pbr.right()
        itemPos[0] = br.right()
    if abs(br.top() - pbr.top()) < abs(br.bottom() - pbr.bottom()):
        anchorPos[1] = 0
        parentPos[1] = pbr.top()
        itemPos[1] = br.top()
    else:
        anchorPos[1] = 1
        parentPos[1] = pbr.bottom()
        itemPos[1] = br.bottom()
    if relative:
        relPos = [(itemPos[0] - pbr.left()) / pbr.width(), (itemPos[1] - pbr.top()) / pbr.height()]
        self.anchor(anchorPos, relPos)
    else:
        offset = itemPos - parentPos
        self.anchor(anchorPos, anchorPos, offset)