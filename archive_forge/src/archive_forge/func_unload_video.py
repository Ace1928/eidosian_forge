import unittest
def unload_video(video, position):
    if position > 0.01:
        video.unload()
        Clock.schedule_once(lambda x: stopTouchApp(), 0.1)