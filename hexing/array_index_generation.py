import numpy as np

from math_utilities import one_hot, x_y_z_sign_convention


def generate_rhombal_array_indices(starting_coordinates, ending_coordinates):
    coordinate_list = list()
    axis_diffs = [ending_coordinate - starting_coordinate for
                  (starting_coordinate, ending_coordinate) in
                  zip(starting_coordinates, ending_coordinates)]
    pos_axis_diffs = np.abs(axis_diffs)
    sign_axis_diffs = np.sign(axis_diffs)
    if np.all(pos_axis_diffs):
        # the two hexes are not aligned along any axis
        long_axis = np.argmax(np.abs(axis_diffs))
        loop_axes = tuple({0, 1, 2} - {long_axis})
        for ii in range(int(pos_axis_diffs[loop_axes[0]] + 1)):
            for jj in range(int(pos_axis_diffs[loop_axes[1]] + 1)):
                coordinate_list.append(tuple(np.array(starting_coordinates).astype(int)
                                             + ii * one_hot(3, loop_axes[0]) * sign_axis_diffs
                                             + jj * one_hot(3, loop_axes[1]) * sign_axis_diffs
                                             + (ii+jj) * one_hot(3, long_axis) * sign_axis_diffs
                                             ))
    else:
        # the two hexes are aligned so the result is a  'flat rhombus' (line)
        for ii in range(max(pos_axis_diffs) + 1):
            coordinate_list.append(tuple(np.array(starting_coordinates) + ii * sign_axis_diffs))
    return coordinate_list


def generate_radial_array_indices(radius, origin=None):
    if origin is None:
        origin = (0, 0, 0)
    coordinate_list = [origin]
    for xx in range(-radius, radius + 1):
        for yy in range(-radius, radius + 1):
            if xx == 0 and yy == 0:
                continue
            zz = - xx - yy
            if abs(zz) > radius:
                continue
            coordinate_list.append((xx + origin[0], yy + origin[1], zz + origin[2]))
    return coordinate_list


def generate_angular_rhombal_array_indices(side_length, orientation, origin=None):
    if origin is None:
        origin = (0, 0, 0)
    coordinate_list = [origin]

    x_constraint = (orientation % 3 - 1) * (2 * (orientation // 3) - 1)
    orientation = (orientation - 1) % 6
    y_constraint = (orientation % 3 - 1) * (2 * (orientation // 3) - 1)
    orientation = (orientation - 1) % 6
    z_constraint = (orientation % 3 - 1) * (2 * (orientation // 3) - 1)

    for xx in range(-side_length, side_length + 1):
        if x_constraint and xx and np.sign(xx) != x_constraint:
            continue
        for yy in range(-side_length, side_length + 1):
            if y_constraint and yy and np.sign(yy) != y_constraint:
                continue
            if xx == 0 and yy == 0:
                continue
            zz = - xx - yy
            if z_constraint and zz and np.sign(zz) != z_constraint:
                continue
            if abs(zz) > side_length:
                continue
            coordinate_list.append((xx + origin[0], yy + origin[1], zz + origin[2]))
    return coordinate_list


def generate_triangular_array_indices(side_length, orientation=None, origin=None):
    if origin is None:
        origin = (0, 0, 0)
    else:
        origin = tuple(origin)
    if orientation is None:
        orientation = 0
    coordinate_list = [origin]
    x_constraint, y_constraint, z_constraint = x_y_z_sign_convention(orientation)

    for xx in range(-side_length, side_length + 1):
        if x_constraint and xx and np.sign(xx) != x_constraint:
            continue
        for yy in range(-side_length, side_length + 1):
            if y_constraint and yy and np.sign(yy) != y_constraint:
                continue
            if xx == 0 and yy == 0:
                continue
            zz = - xx - yy
            if z_constraint and zz and np.sign(zz) != z_constraint:
                continue
            if abs(zz) > side_length:
                continue
            coordinate_list.append((xx + origin[0], yy + origin[1], zz + origin[2]))
    return coordinate_list


def generate_star_array_indices(radius, origin=None):
    if origin is None:
        origin = (0, 0, 0)
    center_hex_coordinates = generate_radial_array_indices(radius, origin)
    edge_triangles = list()
    for orientation in range(6):
        constraints = np.array(x_y_z_sign_convention(orientation))
        majority_sign = sum(constraints)
        triangle_origin = ((constraints != majority_sign) + 1) * constraints * radius
        triangle_coordinates = generate_triangular_array_indices(side_length=radius,
                                                                 orientation=(orientation + 3) % 6,
                                                                 origin=triangle_origin)
        edge_triangles.append(triangle_coordinates)
    coordinate_list = set().union(*(edge_triangles+[center_hex_coordinates]))
    return coordinate_list