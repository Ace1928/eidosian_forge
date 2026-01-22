from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, AmbientLight, Vec4, Vec3
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerPusher, CollisionSphere
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.bullet import BulletSphereShape
import random
import logging

# Configure logging to the most detailed level possible
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Game(ShowBase):
    def __init__(self):
        super().__init__()
        logging.debug("Game initialization started with superclass initialization.")

        # Initialize the window, camera, and other settings with detailed logging
        self.set_background_color(0.1, 0.1, 0.1, 1)
        self.cam.set_pos(
            0, -50, 20
        )  # Adjusted camera position for better scene visibility
        self.cam.look_at(0, 0, 0)
        logging.info("Window, camera, and background color set to dark gray.")

        # Initialize lighting within the game environment with verbose logging
        ambient_light = AmbientLight("ambient_light")
        ambient_light.set_color(Vec4(0.2, 0.2, 0.2, 1))
        ambient_light_node = self.render.attach_new_node(ambient_light)
        self.render.set_light(ambient_light_node)
        logging.info("Ambient light configured with color Vec4(0.2, 0.2, 0.2, 1).")

        point_light = PointLight("point_light")
        point_light.set_color(Vec4(0.9, 0.9, 0.9, 1))
        point_light_node = self.render.attach_new_node(point_light)
        point_light_node.set_pos(10, -20, 20)
        self.render.set_light(point_light_node)
        logging.info(
            "Point light configured with color Vec4(0.9, 0.9, 0.9, 1) at position (10, -20, 20)."
        )

        # Configure the Bullet physics world with detailed logging
        self.world = BulletWorld()
        self.world.set_gravity(Vec3(0, 0, -9.81))
        logging.info(
            "Bullet physics world initialized and gravity set to Vec3(0, 0, -9.81)."
        )

        # Configure collision handling mechanisms with verbose logging
        self.traverser = CollisionTraverser()
        self.pusher = CollisionHandlerPusher()
        logging.info(
            "Collision handling mechanisms initialized with CollisionTraverser and CollisionHandlerPusher."
        )

        # Construct the ground element with detailed logging
        self.create_ground()

        # Construct the player sphere with detailed logging
        self.create_player()

        # Register the player's collision node with the collision traverser with verbose logging
        self.traverser.add_collider(
            self.player_np.find("**/+CollisionNode"), self.pusher
        )
        self.pusher.add_in_pattern("%fn-into-%in")
        self.pusher.add_out_pattern("%fn-out-%in")
        logging.info(
            "Player collision node registered with collision traverser and patterns set."
        )

        # Construct obstacles within the game with detailed logging
        self.create_obstacles()

        # Schedule regular updates with detailed logging
        self.task_mgr.add(self.update, "update")
        logging.info("Update task scheduled with the task manager.")

        # Set the frame rate to 60 frames per second with verbose logging
        globalClock.set_frame_rate(60)
        logging.info("Frame rate set to 60 FPS.")

    def create_ground(self):
        shape = BulletBoxShape(Vec3(10, 10, 1))
        body = BulletRigidBodyNode("Ground")
        body.add_shape(shape)
        np = self.render.attach_new_node(body)
        np.set_pos(0, 0, -2)
        self.world.attach_rigid_body(body)
        logging.debug(
            "Ground element created with BulletBoxShape and positioned at (0, 0, -2)."
        )

    def create_player(self):
        shape = BulletSphereShape(1)
        body = BulletRigidBodyNode("Player")
        body.set_mass(1.0)
        body.add_shape(shape)
        self.player_np = self.render.attach_new_node(body)
        self.player_np.set_pos(0, 0, 2)
        self.world.attach_rigid_body(body)

        # Define the collision sphere for the player with detailed logging
        coll_node = CollisionNode("player")
        coll_node.add_solid(CollisionSphere(0, 0, 0, 1))
        coll_np = self.player_np.attach_new_node(coll_node)
        self.traverser.add_collider(coll_np, self.pusher)
        logging.debug(
            "Player created with BulletSphereShape and collision sphere defined at position (0, 0, 2)."
        )

    def create_obstacles(self):
        for _ in range(10):
            x, y, z = random.uniform(-8, 8), random.uniform(-8, 8), 0
            shape = BulletBoxShape(Vec3(1, 1, 1))
            body = BulletRigidBodyNode("Box")
            body.add_shape(shape)
            np = self.render.attach_new_node(body)
            np.set_pos(x, y, z)
            self.world.attach_rigid_body(body)
        logging.debug(
            "10 obstacles created with BulletBoxShape and randomly positioned within the range (-8, 8)."
        )

    def update(self, task):
        dt = globalClock.get_dt()
        self.world.do_physics(dt)
        logging.debug(f"Physics updated for dt: {dt} seconds.")
        # Debugging: Print the position of a known object each frame
        player_pos = self.player_np.get_pos()
        logging.debug(f"Player position: {player_pos}")
        return task.cont


game = Game()
game.run()
logging.info("Game execution started and main loop running.")
