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
def definePlayerCollisionSphere(self):
    collisionNode = CollisionNode('player')
    collisionNode.addSolid(CollisionSphere(0, 0, 0, 1))
    collisionNodePath = self.playerNodePath.attachNewNode(collisionNode)
    self.traverser.addCollider(collisionNodePath, self.pusher)
    logging.debug('GameEnvironmentInitializer: Player collision sphere defined.')