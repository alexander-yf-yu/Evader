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

WINDOW_HEIGHT = 600
WINDOW_WIDTH = 300
GRAVITY = -250.00

EVADER_DIAMETER = 10
EVADER_SPEED = 30
### Physics collision types
COLLTYPE_BOUNDS = 0
COLLTYPE_BALL = 1
COLLTYPE_EVADE = 2
COLLTYPE_RAY = 3
COLLTYPE_WALL = 4
RAYCAST_PADDING = 5
BALL_DIAMETER = 15
BALL_EVERY = 20

def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y + WINDOW_HEIGHT


def mouse_coll_func(arbiter, space, data):
    """Simple callback that increases the radius of circles touching the mouse"""
    s1, s2 = arbiter.shapes
    s2.unsafe_set_radius(s2.radius + 0.15)
    return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    ### Physics stuff
    space = pymunk.Space()
    space.gravity = 0.0, GRAVITY
    space.damping = 0.8

    # Walls
    static = [
        pymunk.Segment(space.static_body, (0, 0), (0, WINDOW_HEIGHT), 0),
        pymunk.Segment(space.static_body, (WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), 0),
    ]

    for s in static:
        s.collision_type = COLLTYPE_WALL
    space.add(static)

    ## Balls
    balls = []

    #evader
    evader_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    evader_body.position = int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 12)
    evader_shape = pymunk.Circle(evader_body, EVADER_DIAMETER, (0, 0))
    evader_shape.collision_type = COLLTYPE_EVADE
    space.add(evader_body, evader_shape)

    def pre_solve(arb, space, data):
        print('gameover')
        return True

    space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_EVADE).pre_solve = \
        pre_solve

    space.add_collision_handler(COLLTYPE_EVADE, COLLTYPE_WALL).pre_solve = \
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
                evader_body.velocity = evader_body.velocity.x + EVADER_SPEED, evader_body.velocity.y
            elif event.type == KEYDOWN and event.key == K_LEFT:
                evader_body.velocity = evader_body.velocity.x - EVADER_SPEED, evader_body.velocity.y
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

        # evader_body.position = evader_body.position.x + 5, evader_body.position.y

        if pygame.key.get_mods() & KMOD_SHIFT and pygame.mouse.get_pressed()[0]:
            body = pymunk.Body(10, 10)
            p = pygame.mouse.get_pos()
            mouse_pos = Vec2d(p[X], flipy(p[Y]))
            body.position = mouse_pos
            shape = pymunk.Circle(body, 10, (0, 0))
            shape.collision_type = COLLTYPE_BALL
            space.add(body, shape)
            balls.append(shape)

        # Decaying Evader speed
        # if evader_body.velocity.x > 0:
        #     evader_body.velocity = evader_body.velocity.x - EVADER_SPEED, evader_body.velocity.y
        # elif evader_body.velocity.x < 0:
        #     evader_body.velocity = evader_body.velocity.x + EVADER_SPEED, evader_body.velocity.y
        # else:
        #     pass

        ## generate random balls
        if i % BALL_EVERY == 0:
            body = pymunk.Body(10, 10)
            x = random.randint(10, WINDOW_WIDTH - BALL_DIAMETER)
            body.position = x, WINDOW_HEIGHT - 20
            shape = pymunk.Circle(body, BALL_DIAMETER, (0, 0))
            shape.collision_type = COLLTYPE_BALL
            space.add(body, shape)
            balls.append(shape)

        # TODO by AI
        # update evader_body.position
        # evader_body.position = evader_body.position.x - 5, evader_body.position.y
        # OUTPUTS: GO RIGHT, DO NOTHING, GO LEFT

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

        r1 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex, ey + 300), 1, pymunk.ShapeFilter())
        r2 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 20, ey + 295), 1, pymunk.ShapeFilter())
        r3 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 20, ey + 295), 1, pymunk.ShapeFilter())
        r4 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 40, ey + 285), 1, pymunk.ShapeFilter())
        r5 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 40, ey + 285), 1, pymunk.ShapeFilter())
        r6 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex + 60, ey + 260), 1, pymunk.ShapeFilter())
        r7 = space.segment_query_first((ex, ey + EVADER_DIAMETER + RAYCAST_PADDING), (ex - 60, ey + 260), 1, pymunk.ShapeFilter())

        raycasts = [r1, r2, r3, r4, r5, r6, r7]

        for ray in raycasts:
            if ray is not None:
                contact = ray.point
                p1 = int(ex), int(flipy(ey) - EVADER_DIAMETER - RAYCAST_PADDING)
                p2 = int(contact.x), int(flipy(contact.y))
                print(p1, p2)
                # print("LOOP", id(ray.shape))
                pygame.draw.line(screen, THECOLORS["green"], p1, p2, 1)
            else:
                # print("None")
                pass

        for ball in balls[:]:
            v = ball.body.position
            if int(flipy(v.y)) > WINDOW_HEIGHT + 100:
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
        ep = int(ex), int(flipy(ey))
        if not 0 <= ex <= WINDOW_WIDTH:
            pre_solve(None, space, None)
        else:
            pygame.draw.circle(screen, THECOLORS["purple"], ep, int(er), 2)

        # Flip screen
        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))

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
