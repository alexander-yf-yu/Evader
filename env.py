import pygame
from pygame.locals import *
from pygame.color import *
import random

import pymunk
from pymunk import Vec2d

import abc
import tensorflow as tf
import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

# Pygame settings
GRAPHICS = False
WINDOW_HEIGHT = 600
WINDOW_WIDTH = 300
GRAVITY = -250.00

# Evader constants
EVADER_DIAMETER = 10
EVADER_SPEED = 30
EVADER_MOVE_MAG = 1

# Ball constants
BALL_DIAMETER = 10
BALL_EVERY = 20

NUM_RAYS = 7
# Prevent Raycasts from ending early on evader
RAYCAST_PADDING = 2

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
        if GRAPHICS:
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
        self.left_right = 0

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(shape=(NUM_RAYS,),
                                                             dtype=np.float32,
                                                             minimum=0.0,
                                                             maximum=1.0,
                                                             name='observation')

    def move_ev(self, m):
        self.evader_body.position = self.evader_body.position.x + m * EVADER_MOVE_MAG, self.evader_body.position.y

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
        self.left_right = 0
        # Raycasts see nothing when there are no balls,
        # so we initialize as a list of 0s
        new_obs = []
        for _ in range(NUM_RAYS):
            new_obs.append(1.0)

        EvaderEnv.episodes += 1

        print(EvaderEnv.episodes)
        # print(self._observation_spec)

        return ts.restart(np.array(new_obs, dtype=np.float32))

    def _step(self, action):

        if EvaderEnv._episode_ended:
            # The last action ended the episode.
            # Ignore the current action and start a new episode.
            return self.reset()

        if GRAPHICS:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                elif event.type == KEYDOWN and event.key == K_RIGHT:
                    self.evader_body.position = self.evader_body.position.x + EVADER_MOVE_MAG, self.evader_body.position.y
                elif event.type == KEYDOWN and event.key == K_LEFT:
                    self.evader_body.position = self.evader_body.position.x - EVADER_MOVE_MAG, self.evader_body.position.y

        # generate random balls
        if self._state % BALL_EVERY == 0:
            new_body = pymunk.Body(10, 10)
            pos = random.randint(10, WINDOW_WIDTH - BALL_DIAMETER)
            new_body.position = pos, WINDOW_HEIGHT - 20
            new_shape = pymunk.Circle(new_body, BALL_DIAMETER, (0, 0))
            new_shape.collision_type = COLLTYPE_BALL
            self.space.add(new_body, new_shape)
            self.balls.append(new_shape)

        # TODO by AI
        # update evader_body.position
        # OUTPUTS: GO RIGHT, DO NOTHING, GO LEFT
        assert action in [-1, 0, 1]
        self.move_ev(action)
        if action == 1 or action == -1:
            self.left_right += 1

        # Advancing physics
        dt = 1.0 / 60.0
        self.space.step(dt)

        if GRAPHICS:
            self.screen.fill(THECOLORS["white"])

        # RAYCASTING
        ex = self.evader_body.position.x
        ey = self.evader_body.position.y

        r1 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex, ey + 300), 1,
            pymunk.ShapeFilter())
        r2 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 20, ey + 295),
            1, pymunk.ShapeFilter())
        r3 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 20, ey + 295),
            1, pymunk.ShapeFilter())
        r4 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 45, ey + 285),
            1, pymunk.ShapeFilter())
        r5 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 45, ey + 285),
            1, pymunk.ShapeFilter())
        r6 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 70, ey + 260),
            1, pymunk.ShapeFilter())
        r7 = self.space.segment_query_first(
            (ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 70, ey + 260),
            1, pymunk.ShapeFilter())

        raycasts = [r1, r2, r3, r4, r5, r6, r7]
        self.alphas = []

        for ray in raycasts:
            if ray is not None:
                contact = ray.point
                a = float(ray.alpha)
                self.alphas.append(a)
                # self.alphas.append(ray.alpha)
                if GRAPHICS:
                    p1 = int(ex), int(flip_y(ey) - EVADER_DIAMETER - RAYCAST_PADDING)
                    p2 = int(contact.x), int(flip_y(contact.y))
                    pygame.draw.line(self.screen, THECOLORS["green"], p1, p2, 1)
            else:
                self.alphas.append(1.0)

        # print(self._observation_spec)

        for ball in self.balls[:]:
            v = ball.body.position
            if int(flip_y(v.y)) > WINDOW_HEIGHT + 100:
                self.space.remove(ball)
                self.balls.remove(ball)
                # print("remove")
            else:
                if GRAPHICS:
                    r = ball.radius
                    p = int(v.x), int(flip_y(v.y))
                    pygame.draw.circle(self.screen, THECOLORS["blue"], p, int(r), 2)

        if GRAPHICS:
            er = self.evader_shape.radius
            ep = int(ex), int(flip_y(ey))
            pygame.draw.circle(self.screen, THECOLORS["purple"], ep, int(er), 2)

            # Flip screen
            pygame.display.flip()
            self.clock.tick(50)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

        self._state += 1


        # Rewards Calculation
        # reward = frames completed - possible crash - num of left_right moves
        if self._episode_ended:
            reward = float(self._state - 100 - self.left_right / 100)
            return ts.termination(np.array(self.alphas, dtype=np.float32), reward)
        elif self._state >= 500:
            reward = float(self._state - self.left_right / 100)
            return ts.termination(np.array(self.alphas, dtype=np.float32), reward)
        else:
            return ts.transition(np.array(self.alphas, dtype=np.float32), reward=0.0, discount=1.0)

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

    utils.validate_py_environment(env, episodes=9)

    # MAIN LOOP
    while True:
        choice = random.randint(-1, 1)
        env.step(choice)
