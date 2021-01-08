import numpy as np

from constants import defaul_sie


def generate_visual_grid(element_dict, min_x, min_y, max_x, max_y, size=defaul_sie):
    width = max_x - min_x + 1
    height = max_y - min_y + 2
    overbar = u'\u0305'
    element_rectangular_dict = {hex_obj.rectangular_coordinates: hex_obj for hex_obj in element_dict.values()}
    ASPECT_RATIO = 1.7
    full_building_blocks = []
    size_x = int(round(size * ASPECT_RATIO)) + 2
    block_width = size_x + 2 * 1
    for ii in range(size):
        if ii == 0:
            new_line = ' ' * (1 - ii - 1) + "/" + overbar * (size_x + ii * 2) + "\\" + ' ' * (1 - ii - 1)
        else:
            new_line = ' ' * (1 - ii - 1) + "/" + ' ' * (size_x + ii * 2) + "\\" + ' ' * (1 - ii - 1)
        full_building_blocks.append(new_line)
    for ii in reversed(range(size)):
        if ii == 0:
            new_line = ' ' * (1 - ii - 1) + "\\" + '_' * (size_x + ii * 2) + "/" + ' ' * (1 - ii - 1)
        else:
            new_line = ' ' * (1 - ii - 1) + "\\" + ' ' * (size_x + ii * 2) + "/" + ' ' * (1 - ii - 1)
        full_building_blocks.append(new_line)
    # empty_building_block = ' ' * block_width
    lines = []
    for ii in range(height * size * 2):
        new_line = ''
        for jj in range(width):
            # expected_rectangular_coordinates = (jj + min_x, (ii - size * (jj % 2) - ((min_x % 2)) + 1) // (size * 2) + min_y)
            expected_rectangular_coordinates = (jj + min_x, (ii - size * (jj % 2) - size * 2 * ((jj + 1) % 2) * (min_x % 2)) // (size * 2) + min_y)
            offset_index = ii % (size*2)
            if jj == 0:
                new_line += ' ' * (size - 1 - min(offset_index, 2 * size - 1 - offset_index))
            building_block_index = (ii + size * (jj % 2)) % (size * 2)
            if expected_rectangular_coordinates in element_rectangular_dict.keys():
                new_line += full_building_blocks[building_block_index]
            else:
                new_line += ' ' * len(full_building_blocks[building_block_index])
        lines.append(new_line)
    for element in element_dict.values():
        jj, ii = element.rectangular_coordinates
        adjusted_rectangular_coordinates = (jj - min_x,
                                            ii + (jj % 2) * (min_x % 2) - min_y)   # - (min_y % 2))
        x_center = int((size_x + size + 1) * (adjusted_rectangular_coordinates[0] + 0.5)) + 1
        y = int(size * 2 * adjusted_rectangular_coordinates[1] + 1 + (adjusted_rectangular_coordinates[0] % 2 + 1) * size - int((len(element.text) + 1) / 2))
        text_broken_up = element.text
        if len(text_broken_up) > 2 * size:
            text_broken_up = text_broken_up[:2 * size]
        vertical_offset = len(text_broken_up) // 2
        for ind, chars in enumerate(text_broken_up):
            if ind >= size * 2 - 1:
                break
            space = size_x + 2 * min(ind, abs(ind + 1 - size * 2))
            if space < len(chars):
                chars = chars[0:space]
            x = x_center - int(len(chars)/2)
            level = y + ind - vertical_offset
            lines[level] = lines[level][0: x] + chars + lines[level][x + len(chars):]
    lines = remove_empty_lines(lines)
    output = '\n'.join(lines)
    return output


def remove_empty_lines(lines):
    indices_with_material = []
    for ii, line in enumerate(lines):
        if np.any([True for char in line if char != ' ']):
            indices_with_material.append(ii)
    if len(indices_with_material) > 0:
        lines = lines[indices_with_material[0]: indices_with_material[-1] + 1]
    else:
        lines = []
    return lines
