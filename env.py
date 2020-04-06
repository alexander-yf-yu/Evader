import pygame
from pygame.locals import *
from pygame.color import *
import random

import pymunk

import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

# Pygame settings
WINDOW_HEIGHT = 500
WINDOW_WIDTH = 200
GRAVITY = -200.00

# Evader constants
EVADER_DIAMETER = 10
EVADER_MOVE_MAG = 3

# Ball constants
BALL_DIAMETER = 12
BALL_EVERY = 20

# Raycasts

# Prevent Raycasts from ending early on evader
RAYCAST_PADDING = 2

ray_start = [
    [-EVADER_DIAMETER - RAYCAST_PADDING,  RAYCAST_PADDING],
    [-EVADER_DIAMETER + RAYCAST_PADDING, EVADER_DIAMETER - RAYCAST_PADDING],
    [-RAYCAST_PADDING, EVADER_DIAMETER + RAYCAST_PADDING],
    [RAYCAST_PADDING, EVADER_DIAMETER + RAYCAST_PADDING],
    [EVADER_DIAMETER - RAYCAST_PADDING, EVADER_DIAMETER - RAYCAST_PADDING],
    [EVADER_DIAMETER + RAYCAST_PADDING,  RAYCAST_PADDING]
]

ray_end = [
    [-90, 250],
    [-60, 350],
    [-20, 400],
    [20, 400],
    [60, 350],
    [90, 250]
]

NUM_RAYS = len(ray_start)

### Physics collision types
COLLTYPE_BOUNDS = 0
COLLTYPE_BALL = 1
COLLTYPE_EVADE = 2
COLLTYPE_WALL = 3

def flip_y(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + WINDOW_HEIGHT


class EvaderEnv(py_environment.PyEnvironment):

    _episode_ended = False
    graphics = True
    episodes = 0

    @staticmethod
    # Collision Resolver
    def pre_solve(arb, space, data):
        # print('gameover')
        EvaderEnv._episode_ended = True
        return True

    def __init__(self):
        super().__init__()

        # Pygame initializations
        if EvaderEnv.graphics:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space.gravity = 0.0, GRAVITY
        self.space.damping = 0.8

        # Walls
        self.static = [
            pymunk.Segment(self.space.static_body, (0, 0), (0, WINDOW_HEIGHT), 0),
            pymunk.Segment(self.space.static_body, (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), 0)
        ]
        for s in self.static:
            s.collision_type = COLLTYPE_WALL

        self.space.add(self.static)

        # Balls
        self.balls = []

        # Evader
        self.evader_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.evader_body.position = int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 12)
        self.evader_shape = pymunk.Circle(self.evader_body, EVADER_DIAMETER, (0, 0))
        self.evader_shape.collision_type = COLLTYPE_EVADE
        self.space.add(self.evader_body, self.evader_shape)

        # Collision handlers call EvaderEnv.pre_solve when the game is over
        self.space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_EVADE).pre_solve = EvaderEnv.pre_solve
        self.space.add_collision_handler(COLLTYPE_EVADE, COLLTYPE_WALL).pre_solve = EvaderEnv.pre_solve

        # List of evaders in case of parallel training
        self.evaders = []
        # initializing list of outputs
        self.alphas = []
        self._state = 0

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(shape=(NUM_RAYS,),
                                                             dtype=np.float32,
                                                             minimum=0.0,
                                                             maximum=1.0,
                                                             name='observation')

    def move_ev(self, m):
        mag = m - 1
        x = self.evader_body.position.x
        y = self.evader_body.position.y
        self.evader_body.position = x + mag * EVADER_MOVE_MAG, y

    def _reset(self):
        # Put evader back in the middle
        self.evader_body.position = int(WINDOW_WIDTH / 2), int(
                WINDOW_HEIGHT / 12)

        # Get rid of balls in space
        for obj in self.space.shapes:
            if obj.collision_type == COLLTYPE_BALL:
                self.space.remove(obj.body)
                self.space.remove(obj)
        self.balls = []
        EvaderEnv._episode_ended = False

        # Reset state
        self._state = 0
        # Raycasts see nothing when there are no balls,
        # so we initialize as a list of 0s
        new_obs = []
        for _ in range(NUM_RAYS):
            new_obs.append(1.0)

        EvaderEnv.episodes += 1

        # print(EvaderEnv.episodes)

        return ts.restart(np.array(new_obs, dtype=np.float32))

    def _step(self, action):

        if EvaderEnv._episode_ended:
            # The last action ended the episode.
            # Ignore the current action and start a new episode.
            return self.reset()

        if EvaderEnv.graphics:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                elif event.type == KEYDOWN and event.key == K_RIGHT:
                    self.move_ev(2)
                elif event.type == KEYDOWN and event.key == K_LEFT:
                    self.move_ev(0)


        # generate random balls
        if self._state % BALL_EVERY == 0:
            new_body = pymunk.Body(10, 10)
            pos = random.randint(0, WINDOW_WIDTH)
            new_body.position = pos, WINDOW_HEIGHT - BALL_DIAMETER
            new_shape = pymunk.Circle(new_body, BALL_DIAMETER, (0, 0))
            new_shape.collision_type = COLLTYPE_BALL
            self.space.add(new_body, new_shape)
            self.balls.append(new_shape)

        # TODO by AI
        # update evader_body.position
        assert action in [0, 1, 2]
        self.move_ev(action)

        # Advancing physics
        dt = 1.0 / 30.0
        self.space.step(dt)

        if EvaderEnv.graphics:
            self.screen.fill(THECOLORS["white"])

        # RAYCASTING
        ex = self.evader_body.position.x
        ey = self.evader_body.position.y

        self.alphas = []

        for i in range(NUM_RAYS):
            start_x = ex + ray_start[i][0]
            start_y = ey + ray_start[i][1]
            start = start_x, start_y

            end_x = ex + ray_end[i][0]
            end_y = ey + ray_end[i][1]
            end = end_x, end_y

            r = self.space.segment_query_first(start, end, 1, pymunk.ShapeFilter())

            if r is not None:
                contact = r.point
                a = float(r.alpha)
                self.alphas.append(a)
                if EvaderEnv.graphics:
                    p1 = int(start_x), int(flip_y(start_y))
                    p2 = int(contact.x), int(flip_y(contact.y))
                    pygame.draw.line(self.screen, THECOLORS["green"], p1, p2, 1)
            else:
                if EvaderEnv.graphics:
                    p1 = int(start_x), int(flip_y(start_y))
                    p2 = int(end_x), int(flip_y(end_y))
                    pygame.draw.line(self.screen, THECOLORS["green"], p1, p2, 1)
                self.alphas.append(1.0)

        for ball in self.balls[:]:
            v = ball.body.position
            if int(flip_y(v.y)) > WINDOW_HEIGHT + 100:
                self.space.remove(ball)
                self.balls.remove(ball)
                # print("remove")
            else:
                if EvaderEnv.graphics:
                    r = ball.radius
                    p = int(v.x), int(flip_y(v.y))
                    pygame.draw.circle(self.screen, THECOLORS["blue"], p, int(r), 2)

        if EvaderEnv.graphics:
            er = self.evader_shape.radius
            ep = int(ex), int(flip_y(ey))
            pygame.draw.circle(self.screen, THECOLORS["purple"], ep, int(er), 2)

            # Flip screen
            pygame.display.flip()
            self.clock.tick(50)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

        self._state += 1

        space_from_middle = 2 * (abs(int(WINDOW_WIDTH / 2) - ex) / WINDOW_WIDTH)

        if self._episode_ended:
            reward = -100.0
            return ts.termination(np.array(self.alphas,dtype=np.float32),
                                  reward=reward)
        else:
            reward = 1.0 + sum(self.alphas) / NUM_RAYS - space_from_middle
            return ts.transition(np.array(self.alphas, dtype=np.float32),
                                 reward=reward,
                                 discount=1.0)

    def get_info(self):
        pass

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec


if __name__ == "__main__":

    env = EvaderEnv()

    time_step_spec = env.time_step_spec()

    print("discount: " + str(time_step_spec.discount))
    print("steptype: " + str(time_step_spec.step_type))
    print("reward: " + str(time_step_spec.reward))
    print("observation: " + str(time_step_spec.observation))

    print("________________________________________________________")

    time_step = env.reset()

    print("discount: " + str(time_step.discount))
    print("steptype: " + str(time_step.step_type))
    print("reward: " + str(time_step.reward))
    print("observation: " + str(time_step.observation))

    # utils.validate_py_environment(env, episodes=9)

    # MAIN LOOP
    while True:
        random_action = random.randint(0, 2)
        time_step = env.step(1)
        # print("reward: " + str(time_step.reward))
        # print("observation: " + str(time_step.observation))
