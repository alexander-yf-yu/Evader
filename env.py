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


# class EvaderEnv(py_environment.PyEnvironment):
class EvaderEnv(object):

    # Pygame settings
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 300
    GRAVITY = -250.00

    # Evader constants
    EVADER_DIAMETER = 10
    EVADER_SPEED = 30
    EVADER_MOVE_MAG = 5

    # Ball constants
    BALL_DIAMETER = 10
    BALL_EVERY = 20

    # Prevent Raycasts from ending early on evader
    RAYCAST_PADDING = 2

    ### Physics collision types
    COLLTYPE_BOUNDS = 0
    COLLTYPE_BALL = 1
    COLLTYPE_EVADE = 2
    COLLTYPE_WALL = 3

    @classmethod
    def flip_y(cls, y):
        """Small hack to convert chipmunk physics to pygame coordinates"""
        return -y + cls.WINDOW_HEIGHT

    @staticmethod
    def pre_solve(arb, space, data):
        print('gameover')
        space._episode_ended = True
        return True

    def __init__(self):
        super().__init__()

        # Pygame initializations
        pygame.init()
        self.screen = pygame.display.set_mode((EvaderEnv.WINDOW_WIDTH, EvaderEnv.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Physics stuff
        self.space = pymunk.Space()
        self.space.gravity = 0.0, EvaderEnv.GRAVITY
        self.space.damping = 0.8

        # Walls
        self.static = [
            pymunk.Segment(self.space.static_body, (0, 0), (0, EvaderEnv.WINDOW_HEIGHT), 0),
            pymunk.Segment(self.space.static_body, (EvaderEnv.WINDOW_WIDTH, 0), (EvaderEnv.WINDOW_WIDTH, EvaderEnv.WINDOW_HEIGHT), 0)
        ]
        for s in self.static:
            s.collision_type = EvaderEnv.COLLTYPE_WALL

        self.space.add(self.static)

        # Balls
        self.balls = []

        # Evader
        self.evader_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.evader_body.position = int(EvaderEnv.WINDOW_WIDTH / 2), int(EvaderEnv.WINDOW_HEIGHT / 12)
        self.evader_shape = pymunk.Circle(self.evader_body, EvaderEnv.EVADER_DIAMETER, (0, 0))
        self.evader_shape.collision_type = EvaderEnv.COLLTYPE_EVADE
        self.space.add(self.evader_body, self.evader_shape)

        # Collision handlers call EvaderEnv.pre_solve when the game is over
        self.space.add_collision_handler(EvaderEnv.COLLTYPE_BALL, EvaderEnv.COLLTYPE_EVADE).pre_solve = EvaderEnv.pre_solve
        self.space.add_collision_handler(EvaderEnv.COLLTYPE_EVADE, EvaderEnv.COLLTYPE_WALL).pre_solve = EvaderEnv.pre_solve

        self.evaders = []
        self.alphas = []

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def move_ev(self, m):
        self.evader_body.position = self.evader_body.position.x + m * EvaderEnv.EVADER_MOVE_MAG, self.evader_body.position.y

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # Put evader back in the middle
        self.evader_body.position = int(EvaderEnv.WINDOW_WIDTH / 2), int(
                EvaderEnv.WINDOW_HEIGHT / 12)
        # Reset state
        self._state = 0
        # Get rid of balls in space
        for obj in self.space.shapes:
            if obj.collision_type == EvaderEnv.COLLTYPE_BALL:
                self.space.remove(obj.body)
                self.space.remove(obj)
        self.balls = []
        self._episode_ended = False

        # TODO return ts.something
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # generate random balls
        if self._state % EvaderEnv.BALL_EVERY == 0:
            new_body = pymunk.Body(10, 10)
            pos = random.randint(10, EvaderEnv.WINDOW_WIDTH - EvaderEnv.BALL_DIAMETER)
            new_body.position = pos, EvaderEnv.WINDOW_HEIGHT - 20
            new_shape = pymunk.Circle(new_body, EvaderEnv.BALL_DIAMETER, (0, 0))
            new_shape.collision_type = EvaderEnv.COLLTYPE_BALL
            self.space.add(new_body, new_shape)
            self.balls.append(new_shape)

        # TODO by AI
        # update evader_body.position
        # evader_body.position = evader_body.position.x - 1, evader_body.position.y
        # OUTPUTS: GO RIGHT, DO NOTHING, GO LEFT

        # Advancing physics
        dt = 1.0 / 60.0
        self.space.step(dt)
        self.screen.fill(THECOLORS["white"])

        # RAYCASTING
        ex = self.evader_body.position.x
        ey = self.evader_body.position.y

        r1 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex, ey + 300), 1,
            pymunk.ShapeFilter())
        r2 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex + 20, ey + 295),
            1, pymunk.ShapeFilter())
        r3 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex - 20, ey + 295),
            1, pymunk.ShapeFilter())
        r4 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex + 45, ey + 285),
            1, pymunk.ShapeFilter())
        r5 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex - 45, ey + 285),
            1, pymunk.ShapeFilter())
        r6 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex + 70, ey + 260),
            1, pymunk.ShapeFilter())
        r7 = self.space.segment_query_first(
            (ex, ey + EvaderEnv.EVADER_DIAMETER + EvaderEnv.RAYCAST_PADDING), (ex - 70, ey + 260),
            1, pymunk.ShapeFilter())

        raycasts = [r1, r2, r3, r4, r5, r6, r7]

        self.alphas = []

        for ray in raycasts:
            if ray is not None:
                contact = ray.point
                p1 = int(ex), int(
                    EvaderEnv.flip_y(ey) - EvaderEnv.EVADER_DIAMETER - EvaderEnv.RAYCAST_PADDING)
                p2 = int(contact.x), int(EvaderEnv.flip_y(contact.y))
                self.alphas.append(ray.alpha)
                pygame.draw.line(self.screen, THECOLORS["green"], p1, p2, 1)
            else:
                self.alphas.append(1.0)

        # print(self.alphas)
        print(self.__dir__())

        for ball in self.balls[:]:
            v = ball.body.position
            if int(EvaderEnv.flip_y(v.y)) > EvaderEnv.WINDOW_HEIGHT + 100:
                self.space.remove(ball)
                self.balls.remove(ball)
                print("remove")
            else:
                r = ball.radius
                rot = ball.body.rotation_vector
                p = int(v.x), int(EvaderEnv.flip_y(v.y))
                p2 = Vec2d(rot.x, -rot.y) * r * 0.9
                pygame.draw.circle(self.screen, THECOLORS["blue"], p, int(r), 2)
                pygame.draw.line(self.screen, THECOLORS["red"], p, p + p2)

        er = self.evader_shape.radius
        ep = int(ex), int(EvaderEnv.flip_y(ey))
        if not 0 <= ex <= EvaderEnv.WINDOW_WIDTH:
            env.pre_solve(None, self, None)
        else:
            pygame.draw.circle(self.screen, THECOLORS["purple"], ep, int(er), 2)

        # Flip screen
        pygame.display.flip()
        self.clock.tick(50)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

        self._state += 1

        # # Make sure episodes don't go on forever.
        # if action == 1:
        #     self._episode_ended = True
        # elif action == 0:
        #     new_card = np.random.randint(1, 11)
        #     self._state += new_card
        # else:
        #     raise ValueError('`action` should be 0 or 1.')
        #
        # if self._episode_ended or self._state >= 21:
        #     reward = self._state - 21 if self._state <= 21 else -21
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def get_info(self):
        pass


if __name__ == "__main__":

    env = EvaderEnv()

    # MAIN LOOP
    running = True
    while running:
        env._step(1)
