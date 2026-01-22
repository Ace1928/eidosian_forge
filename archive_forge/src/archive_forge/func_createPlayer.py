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
def createPlayer(self):
    shape = BulletSphereShape(1)
    body = BulletRigidBodyNode('Player')
    body.setMass(1.0)
    body.addShape(shape)
    self.playerNodePath = self.render.attachNewNode(body)
    self.playerNodePath.set_pos(0, 0, 2)
    self.playerNodePath.set_color(1, 1, 1, 1)
    self.world.attachRigidBody(body)
    self.definePlayerCollisionSphere()