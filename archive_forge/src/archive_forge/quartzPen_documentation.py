from fontTools.pens.basePen import BasePen
from Quartz.CoreGraphics import CGPathCreateMutable, CGPathMoveToPoint
from Quartz.CoreGraphics import CGPathAddLineToPoint, CGPathAddCurveToPoint
from Quartz.CoreGraphics import CGPathAddQuadCurveToPoint, CGPathCloseSubpath
A pen that creates a CGPath

    Parameters
    - path: an optional CGPath to add to
    - xform: an optional CGAffineTransform to apply to the path
    