import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def computeRollover(self, currentTime):
    """
        Work out the rollover time based on the specified time.
        """
    result = currentTime + self.interval
    if self.when == 'MIDNIGHT' or self.when.startswith('W'):
        if self.utc:
            t = time.gmtime(currentTime)
        else:
            t = time.localtime(currentTime)
        currentHour = t[3]
        currentMinute = t[4]
        currentSecond = t[5]
        currentDay = t[6]
        if self.atTime is None:
            rotate_ts = _MIDNIGHT
        else:
            rotate_ts = (self.atTime.hour * 60 + self.atTime.minute) * 60 + self.atTime.second
        r = rotate_ts - ((currentHour * 60 + currentMinute) * 60 + currentSecond)
        if r < 0:
            r += _MIDNIGHT
            currentDay = (currentDay + 1) % 7
        result = currentTime + r
        if self.when.startswith('W'):
            day = currentDay
            if day != self.dayOfWeek:
                if day < self.dayOfWeek:
                    daysToWait = self.dayOfWeek - day
                else:
                    daysToWait = 6 - day + self.dayOfWeek + 1
                newRolloverAt = result + daysToWait * (60 * 60 * 24)
                if not self.utc:
                    dstNow = t[-1]
                    dstAtRollover = time.localtime(newRolloverAt)[-1]
                    if dstNow != dstAtRollover:
                        if not dstNow:
                            addend = -3600
                        else:
                            addend = 3600
                        newRolloverAt += addend
                result = newRolloverAt
    return result