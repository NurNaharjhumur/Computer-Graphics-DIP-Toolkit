import cv2
import numpy as np

def rotate_image(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if img is not None else img

def scale_image(img):
    return cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR) if img is not None else img

def reflect_image_horizontal(img):
    return cv2.flip(img, 1) if img is not None else img

def shear_image_x(img):
    if img is None:
        return img
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0.2, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (int(cols * 1.2), rows))

def launch_opengl_window():
    from OpenGL.GL import glClear, glClearColor, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glRotatef, glBegin, glEnd, glVertex3f, glColor3f, GL_QUADS, glEnable, GL_DEPTH_TEST
    from OpenGL.GLU import gluPerspective
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("OpenGL 3D Cube Viewer")
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    pygame.draw.rect(pygame.display.get_surface(), (0,0,0), (0,0,800,600))
    import OpenGL.GL as gl
    gl.glTranslatef(0.0, 0.0, -5)

    vertices = [
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ]
    surfaces = [
        (0,1,2,3), (3,2,6,7), (7,6,5,4),
        (4,5,1,0), (1,5,6,2), (4,0,3,7)
    ]
    colors = [
        (1,0,0), (0,1,0), (0,0,1),
        (1,1,0), (1,0,1), (0,1,1)
    ]

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glRotatef(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glBegin(GL_QUADS)
        for i, surface in enumerate(surfaces):
            glColor3f(*colors[i])
            for vertex in surface:
                glVertex3f(*vertices[vertex])
        glEnd()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
