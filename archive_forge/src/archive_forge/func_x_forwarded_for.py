from tensorboard import auth as auth_lib
@property
def x_forwarded_for(self):
    return self._x_forwarded_for