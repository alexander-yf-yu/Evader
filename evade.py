"""This example lets you dynamically create static walls and dynamic balls

"""
__docformat__ = "reStructuredText"

import pygame
from pygame.locals import *
from pygame.color import *
import random

import pymunk
from pymunk import Vec2d

X, Y = 0, 1

EVADER_DIAMETER = 10
### Physics collision types
COLLTYPE_BOUNDS = 0
COLLTYPE_BALL = 1
COLLTYPE_EVADE = 2
COLLTYPE_RAY = 3
RAYCAST_PADDING = 5


def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + 600


def mouse_coll_func(arbiter, space, data):
    """Simple callback that increases the radius of circles touching the mouse"""
    s1, s2 = arbiter.shapes
    s2.unsafe_set_radius(s2.radius + 0.15)
    return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    running = True

    ### Physics stuff
    space = pymunk.Space()
    space.gravity = 0.0, -900.0

    ## Balls
    balls = []

    #evader
    evader_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    evader_body.position = 300, 50
    evader_shape = pymunk.Circle(evader_body, EVADER_DIAMETER, (0, 0))
    evader_shape.collision_type = COLLTYPE_EVADE
    space.add(evader_body, evader_shape)

    def pre_solve(arb, space, data):
        print('gameover')
        return True

    space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_EVADE).pre_solve = \
        pre_solve

    evaders = []
    run_physics = True

    i = 0
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(screen, "balls_and_lines.png")
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                print(evader_body.is_sleeping)
                evader_body.position = evader_body.position.x + 5, evader_body.position.y
            elif event.type == KEYDOWN and event.key == K_LEFT:
                evader_body.position = evader_body.position.x - 5, evader_body.position.y
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                p = event.pos[X], flipy(event.pos[Y])
                body = pymunk.Body(10, 100)
                body.position = p
                shape = pymunk.Circle(body, 10, (0, 0))
                shape.friction = 0.5
                shape.collision_type = COLLTYPE_BALL
                space.add(body, shape)
                balls.append(shape)

            elif event.type == KEYDOWN and event.key == K_SPACE:
                run_physics = not run_physics

        if pygame.key.get_mods() & KMOD_SHIFT and pygame.mouse.get_pressed()[0]:
            body = pymunk.Body(10, 10)
            p = pygame.mouse.get_pos()
            mouse_pos = Vec2d(p[X], flipy(p[Y]))
            body.position = mouse_pos
            shape = pymunk.Circle(body, 10, (0, 0))
            shape.collision_type = COLLTYPE_BALL
            space.add(body, shape)
            balls.append(shape)

        ## generate random balls
        if i % 10 == 0:
            body = pymunk.Body(10, 10)
            x = random.randint(10, 590)
            body.position = x, 580
            shape = pymunk.Circle(body, 30, (0, 0))
            shape.collision_type = COLLTYPE_BALL
            space.add(body, shape)
            balls.append(shape)

        # TODO by AI
        # update evader_body.position
        # evader_body.position = ???

        ### Update physics
        if run_physics:
            dt = 1.0 / 60.0
            space.step(dt)


        ### Draw stuff
        screen.fill(THECOLORS["white"])

        # RAYCASTING
        # create line segments that extend from the evader
        # make custom collision handlers between segments and balls

        ex = evader_body.position.x
        ey = evader_body.position.y

        r1 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex, ey + 200), 1, pymunk.ShapeFilter())
        r2 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 20, ey + 190), 1, pymunk.ShapeFilter())
        r3 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 20, ey + 190), 1, pymunk.ShapeFilter())
        r4 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 40, ey + 175), 1, pymunk.ShapeFilter())
        r5 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 40, ey + 175), 1, pymunk.ShapeFilter())

        raycasts = [r1, r2, r3, r4, r5]

        for ray in raycasts:
            if ray is not None:
                contact = ray.point
                # print(ray.shape, ray.shape.body)
                # line = pymunk.Segment(space.static_body, (0, 0), contact, 1)
                # line.body.position = ex, ey
                # print("hit")
                # print(contact)
                p1 = int(ex), int(flipy(ey) - EVADER_DIAMETER - RAYCAST_PADDING)
                p2 = int(contact.x), int(flipy(contact.y))
                print(p1, p2)
                pygame.draw.line(screen, THECOLORS["green"], p1, p2, 1)
            else:
                # print("None")
                pass

        for ball in balls[:]:
            v = ball.body.position
            if int(flipy(v.y)) > 700:
                space.remove(ball)
                balls.remove(ball)
                print("remove")
            else:
                r = ball.radius
                rot = ball.body.rotation_vector
                p = int(v.x), int(flipy(v.y))
                p2 = Vec2d(rot.x, -rot.y) * r * 0.9
                pygame.draw.circle(screen, THECOLORS["blue"], p, int(r), 2)
                pygame.draw.line(screen, THECOLORS["red"], p, p + p2)

        er = evader_shape.radius
        ep = int(evader_shape.body.position.x), int(flipy(evader_shape.body.position.y))
        pygame.draw.circle(screen, THECOLORS["purple"], ep, int(er), 2)

        # Flip screen
        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))

        # print(i)
        i += 1

if __name__ == '__main__':
    doprof = 0
    if not doprof:
        main()
    else:
        import cProfile, pstats

        prof = cProfile.run("main()", "profile.prof")
        stats = pstats.Stats("profile.prof")
        stats.strip_dirs()
        stats.sort_stats('cumulative', 'time', 'calls')
        stats.print_stats(30)
