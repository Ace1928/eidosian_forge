from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
from panda3d.core import MouseWatcher, ModifierButtons, PGMouseWatcherBackground
import random
import logging

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class GameEnvironmentInitializer(ShowBase):
    def __init__(self):
        super().__init__()
        logging.debug("GameEnvironmentInitializer: Superclass initialization complete.")

        self.configurePhysicsWorld()
        self.initializeCollisionHandlingSystem()  # Initialize collision system before constructing elements
        self.constructEnvironmentalElements()  # Now safe to construct elements that depend on the collision system
        self.configureWindowCameraAndBackground()  # Now safe to reparent the camera
        self.initializeLightingSystem()
        self.scheduleRegularUpdates()
        self.setFrameRate()
        self.enablePlayerMovement()
        self.bindMouseToCamera()

    def configureWindowCameraAndBackground(self):
        self.setBackgroundColor(0.8, 0.8, 0.8, 1)  # Light background color
        self.disableMouse()  # Disable default mouse control
        self.camera.reparent_to(self.playerNodePath)  # Parent camera to player
        self.camera.set_pos(0, 0, 2)  # Set camera position to player's head level
        self.camera.look_at(0, 0, 0)  # Initial camera orientation
        logging.info(
            "GameEnvironmentInitializer: Window, camera, and background color configured."
        )

    def initializeLightingSystem(self):
        self.configureAmbientLight()
        self.configurePointLight()

    def configureAmbientLight(self):
        ambientLight = AmbientLight("ambient_light")
        ambientLight.setColor(Vec4(0.6, 0.6, 0.6, 1))  # Lighter ambient light
        ambientLightNode = self.render.attachNewNode(ambientLight)
        self.render.setLight(ambientLightNode)
        logging.info("GameEnvironmentInitializer: Ambient light configured.")

    def configurePointLight(self):
        pointLight = PointLight("point_light")
        pointLight.setColor(Vec4(0.9, 0.9, 0.9, 1))
        pointLightNode = self.render.attachNewNode(pointLight)
        pointLightNode.set_pos(10, -20, 20)
        self.render.setLight(pointLightNode)
        logging.info("GameEnvironmentInitializer: Point light configured.")

    def configurePhysicsWorld(self):
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        logging.info(
            "GameEnvironmentInitializer: Physics world configured with gravity."
        )

    def initializeCollisionHandlingSystem(self):
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        logging.info(
            "GameEnvironmentInitializer: Collision handling system initialized."
        )

    def constructEnvironmentalElements(self):
        self.createGround()
        self.createPlayer()
        self.createObstacles()

    def createGround(self):
        shape = BulletBoxShape(Vec3(10, 10, 1))
        body = BulletRigidBodyNode("Ground")
        body.addShape(shape)
        nodePath = self.render.attachNewNode(body)
        nodePath.set_pos(0, 0, -2)
        nodePath.set_color(0, 1, 0, 1)  # Green ground
        self.world.attachRigidBody(body)
        logging.debug(
            "GameEnvironmentInitializer: Ground element created and positioned."
        )

    def createPlayer(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode("Player")
        body.setMass(1.0)
        body.addShape(shape)
        self.playerNodePath = self.render.attachNewNode(body)
        self.playerNodePath.set_pos(0, 0, 2)
        self.playerNodePath.set_color(0, 0, 0, 1)  # Black player
        self.world.attachRigidBody(body)
        self.definePlayerCollisionSphere()

    def definePlayerCollisionSphere(self):
        collisionNode = CollisionNode("player")
        collisionNode.addSolid(CollisionSphere(0, 0, 0, 1))
        collisionNodePath = self.playerNodePath.attachNewNode(collisionNode)
        self.traverser.addCollider(collisionNodePath, self.pusher)
        logging.debug("GameEnvironmentInitializer: Player collision sphere defined.")

    def createObstacles(self):
        for _ in range(10):
            x, y, z = random.uniform(-8, 8), random.uniform(-8, 8), 0
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode("Obstacle")
            body.addShape(shape)
            nodePath = self.render.attachNewNode(body)
            nodePath.set_pos(x, y, z)
            self.world.attachRigidBody(body)
        logging.debug("GameEnvironmentInitializer: Obstacles created and positioned.")

    def scheduleRegularUpdates(self):
        self.taskMgr.add(self.updatePhysicsAndLogging, "update")
        logging.info("GameEnvironmentInitializer: Regular updates scheduled.")

    def setFrameRate(self):
        globalClock.setFrameRate(60)
        logging.info("GameEnvironmentInitializer: Frame rate set to 60 FPS.")

    def enablePlayerMovement(self):
        self.accept("arrow_up", self.movePlayer, [Vec3(0, 2, 0)])
        self.accept("arrow_down", self.movePlayer, [Vec3(0, -2, 0)])
        self.accept("arrow_left", self.movePlayer, [Vec3(-2, 0, 0)])
        self.accept("arrow_right", self.movePlayer, [Vec3(2, 0, 0)])
        logging.info("GameEnvironmentInitializer: Player movement enabled.")

    def bindMouseToCamera(self):
        self.mouseWatcherNode = MouseWatcher()
        self.mouseWatcherNode.set_modifier_buttons(ModifierButtons())
        self.mouseWatcherNode.get_display_region(self.win.get_display_region(0))
        self.mouseWatcherNode.get_enter_pattern("mouse-enter-%r")
        self.mouseWatcherNode.get_leave_pattern("mouse-leave-%r")
        self.mouseWatcherNode.get_within_pattern("mouse-within-%r")
        self.mouseWatcherNode.get_without_pattern("mouse-without-%r")
        logging.info("GameEnvironmentInitializer: Mouse to camera binding configured.")

    def movePlayer(self, direction):
        self.playerNodePath.setPos(self.playerNodePath.getPos() + direction)
        logging.debug(
            f"GameEnvironmentInitializer: Player moved to {self.playerNodePath.getPos()}"
        )

    def updatePhysicsAndLogging(self, task):
        deltaTime = globalClock.get_dt()
        self.world.doPhysics(deltaTime)
        playerPosition = self.playerNodePath.get_pos()
        logging.debug(
            f"GameEnvironmentInitializer: Physics updated for deltaTime: {deltaTime}, Player position: {playerPosition}"
        )
        return task.cont


gameEnvironment = GameEnvironmentInitializer()
gameEnvironment.run()
logging.info(
    "GameEnvironmentInitializer: Game execution started and main loop running."
)
