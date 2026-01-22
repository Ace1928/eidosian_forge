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
def configurePointLight(self):
    pointLight = PointLight('point_light')
    pointLight.setColor(Vec4(1, 1, 1, 1))
    pointLightNode = self.render.attachNewNode(pointLight)
    pointLightNode.set_pos(5, -15, 10)
    self.render.setLight(pointLightNode)
    logging.info('GameEnvironmentInitializer: Point light configured.')