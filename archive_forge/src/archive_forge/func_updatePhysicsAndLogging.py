from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3, DirectionalLight
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import MouseWatcher, ModifierButtons, PGMouseWatcherBackground
from panda3d.core import WindowProperties
import random
import logging
def updatePhysicsAndLogging(self, task):
    deltaTime = globalClock.get_dt()
    self.world.doPhysics(deltaTime)
    playerPosition = self.playerNodePath.get_pos()
    logging.debug(f'GameEnvironmentInitializer: Physics updated for deltaTime: {deltaTime}, Player position: {playerPosition}')
    return task.cont