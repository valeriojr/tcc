import glfw
import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

import camera


class PyOpenGLRenderer:
    def __init__(self, window_width, window_height, fov, window_title=''):
        self.fov = fov
        self.camera = camera.Camera()

        glfw.init()
        self.window = glfw.create_window(window_width, window_height, window_title, monitor=None, share=None)
        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)
        glutInit()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_SCISSOR_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.529, 0.808, 0.922, 1.0)
        glReadBuffer(GL_FRONT)

        width, height = glfw.get_window_size(self.window)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, width / height, 0.001, 100.0)

    def clear_screen(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def get_window_size(self):
        return glfw.get_window_size(self.window)

    def set_view(self, x, y, width, height):
        glScissor(x, y, width, height)
        glViewport(x, y, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, width / height, 0.001, 100.0)

    def pre_render(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def update(self):
        glfw.swap_buffers(self.window)

    def load_texture(self, path):
        image = Image.open(path)
        width, height = image.size
        data = numpy.array(image.getdata(), dtype=numpy.uint8)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

        return texture