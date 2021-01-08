import numpy as np
import math


def translation(cubic_coordinates, direction, distance):
    """
    Gives the cubic coordinates of the hex gotten by translating a distance in a direction
    direction = 0 is downward in the grid, increasing rotates counter-clockwise
    """
    x, y, z = cubic_coordinates
    if direction == 0 or direction == 1:
        y -= distance
    if direction == 1 or direction == 2:
        x += distance
    if direction == 2 or direction == 3:
        z -= distance
    if direction == 3 or direction == 4:
        y += distance
    if direction == 4 or direction == 5:
        x -= distance
    if direction == 5 or direction == 0:
        z += distance
    return (x, y, z)


def distance(cubic_coordinates_1, cubic_coordinates_2):
    delta_x = abs(cubic_coordinates_1[0] - cubic_coordinates_2[0])
    delta_y = abs(cubic_coordinates_1[1] - cubic_coordinates_2[1])
    delta_z = abs(cubic_coordinates_1[2] - cubic_coordinates_2[2])
    return sum([delta_x, delta_y, delta_z]) - max(delta_x, delta_y, delta_z)


def one_hot(length, index):
    output = np.zeros(length).astype(int)
    output[index] = 1
    return output


def x_y_z_sign_convention(orientation):
    x_constraint = 2 * (((orientation - 0) % 6) // 3) - 1
    y_constraint = 2 * (((orientation + 2) % 6) // 3) - 1
    z_constraint = 2 * (((orientation - 2) % 6) // 3) - 1
    return x_constraint, y_constraint, z_constraint


def convert_cubic_coordinates_to_rectangular(cubic_coordinates):
    rectangular_coordinates = (cubic_coordinates[0], int(cubic_coordinates[2] + (cubic_coordinates[0] - (cubic_coordinates[0] % 2)) / 2))
    return rectangular_coordinates


def convert_rectangular_coordinates_to_cubic(rectangular_coordinates):
    cubic_coordinates = (rectangular_coordinates[0], 0, rectangular_coordinates[1] - (rectangular_coordinates[0] - (rectangular_coordinates[0] % 1)) / 2)
    cubic_coordinates = (cubic_coordinates[0], -cubic_coordinates[0] - cubic_coordinates[2], cubic_coordinates[2])
    return cubic_coordinates


def convert_cubic_coordinates_to_polar(cubic_coordinates):
    positional_rectangular_x = cubic_coordinates[0] * math.sin(math.pi/3)
    positional_rectangular_y = (cubic_coordinates[1] - cubic_coordinates[2]) / 2
    radius = math.sqrt(positional_rectangular_x ** 2 + positional_rectangular_y ** 2)
    if positional_rectangular_x == 0:
        angle = math.pi/2 * np.sign(positional_rectangular_y)
    elif positional_rectangular_y == 0:
        angle = math.pi * (positional_rectangular_x < 0)
    else:
        angle = math.atan(positional_rectangular_y / positional_rectangular_x)
    angle %= math.pi*2
    return radius, angle


def complete_cubic_coordinates(incomplete_cubic_coordinates):
    if len(incomplete_cubic_coordinates) == 2 or incomplete_cubic_coordinates[2] is None:
        cubic_coordinates = (incomplete_cubic_coordinates[0],
                             incomplete_cubic_coordinates[1],
                             -incomplete_cubic_coordinates[0] - incomplete_cubic_coordinates[1])
    elif incomplete_cubic_coordinates[0] is None:
        cubic_coordinates = (-incomplete_cubic_coordinates[1] - incomplete_cubic_coordinates[2],
                             incomplete_cubic_coordinates[1],
                             incomplete_cubic_coordinates[2],)
    elif incomplete_cubic_coordinates[1] is None:
        cubic_coordinates = (incomplete_cubic_coordinates[0],
                             -incomplete_cubic_coordinates[0] - incomplete_cubic_coordinates[2],
                             incomplete_cubic_coordinates[2],)
    else:
        cubic_coordinates = incomplete_cubic_coordinates
    return cubic_coordinates
