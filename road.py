import random

from OpenGL.GL import *
from OpenGL.GLUT import *


class Road:
    def __init__(self, renderer):
        self.buildings = [] * 10
        for i in range(20):
            width = 0.2 + 0.3 * random.random()
            height = 0.2 + 0.3 * random.random()
            depth = 0.2 + 0.3 * random.random()
            x = 0.5 * (1.0 + width)
            y = 0.5 * height
            z = -0.5 * i
            c = 0.5 + 0.25 * random.random()
            self.buildings.append((x, y, z, width, height, depth, c, c, 0.5 + c))
            c = 0.5 + 0.25 * random.random()
            self.buildings.append((-x, y, z, width, height, depth, c, c, 0.5 + c))

        self.trees = []
        for i in range(10):
            x = 0.3 if (i % 2 == 0) else -0.3
            y = 0.0
            z = -2 * i + 2
            height = 0.2 + 0.1 * random.random()
            radius = 0.1 + 0.5 * height * random.random()
            self.trees.append((x, y, z, height, radius))



    def __render_tree(self, x, y, z, height, radius):
        glPushMatrix()
        glTranslatef(x, 0.5 * height, z)
        glPushMatrix()
        glScalef(0.04, height, 0.04)
        glColor3f(0.25, 0.1, 0.1)
        glutSolidCube(1.0)
        glPopMatrix()

        glTranslatef(0.0, 0.15, 0.0)
        glColor3f(0.3, 0.5, 0.2)
        glutSolidSphere(radius, 8, 8)

        glPopMatrix()

    def render(self):
        for building in self.buildings:
            glPushMatrix()

            x, y, z, width, height, depth, r, g, b = building
            glTranslatef(x, y, z)
            glScalef(width, height, depth)

            glColor3f(r, g, b)
            glutSolidCube(1.0)

            glPopMatrix()

        for tree in self.trees:
            self.__render_tree(*tree)
