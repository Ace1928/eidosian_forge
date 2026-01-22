from collections import deque
from kivy.logger import Logger
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
def process_frame(self, frame):
    events = []
    touches = self.touches
    available_uid = []
    for hand in frame.hands:
        for finger in hand.fingers:
            uid = '{0}:{1}'.format(hand.id, finger.id)
            available_uid.append(uid)
            position = finger.tip_position
            args = (position.x, position.y, position.z)
            if uid not in touches:
                touch = LeapFingerEvent(self.device, uid, args)
                events.append(('begin', touch))
                touches[uid] = touch
            else:
                touch = touches[uid]
                touch.move(args)
                events.append(('update', touch))
    for key in list(touches.keys())[:]:
        if key not in available_uid:
            events.append(('end', touches[key]))
            del touches[key]
    return events