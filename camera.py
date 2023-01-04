from OpenGL.GLU import gluLookAt


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.center_z = -1.0

    def look_at(self, x, y, z):
        self.center_x = x
        self.center_y = y
        self.center_z = z

    def set_position(self, x, y, z, keep_focus=False):
        if not keep_focus:
            self.center_x = x + self.center_x - self.x
            self.center_y = y + self.center_y - self.y
            self.center_z = z + self.center_z - self.z

        self.x = x
        self.y = y
        self.z = z

    def use(self):
        gluLookAt(self.x, self.y, self.z, self.center_x, self.center_y, self.center_z, 0.0, 1.0, 0.0)