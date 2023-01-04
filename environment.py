import math
import time

import cv2
import numpy
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GLU import gluLookAt

import road

DEFAULT_ENV_SETTINGS = {
    'window_width': 640,
    'window_height': 480,
    'window_title': 'Maquete',
    'vehicle_camera_width': 96,
    'vehicle_camera_height': 96,
    'vehicle_camera_channels': 3,
    'vehicle_camera_margin_x': 16,
    'vehicle_camera_margin_y': 16,
}

DEFAULT_VEHICLE_SETTINGS = {
    'axle_distance': 0.05,
    'axle_length': 0.05,
    'wheel_radius': 0.025,
    'top_speed': 0.05,
    'steering_range': math.radians(75.0),
}


class Vehicle:
    def __init__(self, position, angle, **kwargs):
        settings = DEFAULT_VEHICLE_SETTINGS | kwargs

        self.axle_distance = settings['axle_distance']
        self.axle_length = settings['axle_length']
        self.wheel_radius = settings['wheel_radius']
        self.top_speed = settings['top_speed']
        self.steering_range = settings['steering_range']

        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.angle = angle

    def render(self):
        glTranslatef(self.x, self.y, self.z)
        glRotatef(math.degrees(self.angle), 0.0, 1.0, 0.0)
        glPushMatrix()
        glTranslatef(0.0, self.wheel_radius, -0.5 * self.axle_distance)
        glScale(self.axle_length, 0.05, self.axle_distance)
        glColor3f(1.0, 1.0, 0.0)
        glutSolidCube(1.0)
        glTranslatef(0.0, 0.5, -0.5)
        glScalef(1.0, 0.5, 0.5)
        glColor3f(0.0, 0.0, 0.0)
        glutSolidCube(1.0)
        glPopMatrix()


class Environment:
    initial_pos_range = numpy.array([
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0]
    ])
    initial_angle_range = numpy.radians(numpy.array([-45.0, 45.0]))

    def __init__(self, agents, renderer, **kwargs):
        self.agents = agents
        self.renderer = renderer
        self.position = numpy.zeros((agents, 3))
        self.angle = numpy.zeros(agents)
        self.state = numpy.zeros((agents, 96, 96, 3))
        self.done = numpy.full(agents, False)
        self.rewards = numpy.zeros(agents)

        self.road_texture = renderer.load_texture('res/Textures/road.png')
        self.grass_texture = renderer.load_texture('res/Textures/grass.png')

    def render_road(self, road_width, road_length):
        glBindTexture(GL_TEXTURE_2D, self.grass_texture)
        # glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_QUADS)
        glTexCoord(0.0, 200.0)
        glVertex3f(-100.0, 0.0, 100.0)

        glTexCoord(200.0, 200.0)
        glVertex3f(100.0, 0.0, 100.0)

        glTexCoord(200.0, 0.0)
        glVertex3f(100.0, 0.0, -100.0)

        glTexCoord(0.0, 0.0)
        glVertex3f(-100.0, 0.0, -100.0)

        glEnd()

        glBindTexture(GL_TEXTURE_2D, self.road_texture)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUAD_STRIP)
        glTexCoord(0.0, 1.0)
        glVertex3f(-0.5 * road_width, 0.001, 0.5 * road_length)

        glTexCoord(0.0, 0.0)
        glVertex3f(0.5 * road_width, 0.001, 0.5 * road_length)

        glTexCoord(-20.0, 1.0)
        glVertex3f(-0.5 * road_width, 0.001, -0.5 * road_length)

        glTexCoord(-20.0, 0.0)
        glVertex3f(0.5 * road_width, 0.001, -0.5 * road_length)
        glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)

    def get_state(self) -> numpy.ndarray:
        agent_count = 0
        width, height = self.renderer.get_window_size()
        rows, columns = height // 96, width // 96
        frame_width = 96 * columns
        frame_height = 96 * rows
        frame_size = (frame_height, frame_width)
        while agent_count < self.agents:

            agent_offset = agent_count
            for i in range(rows):
                for j in range(columns):
                    agent = agent_offset + i * columns + j
                    if agent >= self.agents:
                        break
                    self.renderer.set_view(96 * j, 96 * i, 96, 96)
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    self.renderer.camera.set_position(
                        self.position[agent, 0],
                        2 * DEFAULT_VEHICLE_SETTINGS['wheel_radius'],
                        self.position[agent, 2] - DEFAULT_VEHICLE_SETTINGS['axle_distance']
                    )
                    self.renderer.camera.look_at(
                        self.position[agent, 0] + numpy.cos(self.angle[agent] + 0.5 * numpy.pi),
                        self.position[agent, 1],
                        self.position[agent, 2] - numpy.sin(self.angle[agent] + 0.5 * numpy.pi)
                    )
                    self.renderer.camera.use()
                    self.render_road(0.4, 10.0)
                    agent_count += 1

            self.renderer.update()
            #
            # r = numpy.frombuffer(glReadPixels(0, 0, frame_width, frame_height, GL_RED, GL_FLOAT),
            #                      dtype=numpy.float32).reshape(frame_size)
            # g = numpy.frombuffer(glReadPixels(0, 0, frame_width, frame_height, GL_GREEN, GL_FLOAT),
            #                      dtype=numpy.float32).reshape(frame_size)
            # b = numpy.frombuffer(glReadPixels(0, 0, frame_width, frame_height, GL_BLUE, GL_FLOAT),
            #                      dtype=numpy.float32).reshape(frame_size)
            # state = numpy.dstack((b, g, r))
            state = numpy.frombuffer(glReadPixels(0, 0, frame_width, frame_height, GL_RGB, GL_FLOAT),
                                     dtype=numpy.float32).reshape((*frame_size, 3))
            state = numpy.flip(state, axis=0)

            for i in range(rows):
                for j in range(columns):
                    agent = agent_offset + i * columns + j
                    if agent >= self.agents:
                        break
                    row_start, row_end = frame_height - (i + 1) * 96, frame_height - i * 96
                    col_start, col_end = frame_width - (j + 1) * 96, frame_width - j * 96
                    self.state[agent] = state[row_start:row_end, col_start:col_end, :]

        return self.state

    def render(self):
        width, height = self.renderer.get_window_size()
        self.renderer.set_view(0, 0, width, height)

    def reset(self) -> numpy.ndarray:
        self.rewards = numpy.zeros(self.agents)
        self.done = numpy.full(self.agents, False)
        self.position = self.initial_pos_range[0] + (numpy.random.random((self.agents, 3)) * self.initial_pos_range[1])
        self.angle = self.initial_angle_range[0] + (numpy.random.random((self.agents,)) * self.initial_angle_range[1])

        return self.get_state()

    def step(self, action, time_step, final_step=False) -> tuple:
        dt = time_step / 1000.0

        speed = action[0]
        steering = action[1]

        speed = (speed - 2) / 2
        steering = (steering - 2) / 2

        self.angle += speed * steering * math.radians(45.0) * dt
        self.position[self.done == False, 0] += (speed * 0.1 * numpy.cos(self.angle + 0.5 * numpy.pi) * dt)[self.done == False]
        self.position[self.done == False, 2] -= (speed * 0.1 * numpy.sin(self.angle + 0.5 * numpy.pi) * dt)[self.done == False]

        # self.done = self.position[:, 2] < -0.025

        self.rewards = -100 * self.position[:, 2] if final_step else numpy.zeros(self.agents)
        self.rewards[numpy.abs(self.position[:, 0]) > 0.15] = -1.0
        # self.rewards[self.done == True] = 0.0

        return self.done, self.rewards, self.get_state()


if __name__ == '__main__':
    import glfw

    import rendering

    renderer = rendering.PyOpenGLRenderer(1280, 720, 60)
    env = Environment(200, renderer)

    env.reset()

    while not glfw.window_should_close(renderer.window):
        t = time.time()
        done, r, state = env.step(numpy.ones((env.agents, 2, 2)), 128)

        # env.render()
        # glfw.swap_buffers(renderer.window)
        glfw.poll_events()
        print(f'{time.time() - t}ms per frame')
