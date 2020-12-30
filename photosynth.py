from hexing import *
import numpy as np
import random
import itertools

move_names = ['plant', 'special_plant', 'pass', 'upgrade', 'collect', 'purchase']

class Tree:
    """
    Tree or seed object
    """
    def __init__(self, player, height):
        self.player = player
        self.height = height

    def __deepcopy__(self, memodict={}):
        new = Tree(self.player, self.height)

    def __str__(self):
        return f'P{str(self.player)}\nH: {str(self.height)}'

    def __eq__(self, other):
        if type(other) != Tree:
            return False
        return self.player == other.player and self.height == other.height

false_array = generate_radial_hex_array(3, False)


def one_hot(k, n):
    output = np.zeros(n)
    output[k] = 1
    return output

def belongs_to_player(tree, player):
    if tree is None:
        return False
    return tree.player == player


# board = generate_radial_hex_array(3)
# board[(-2, -1, 3)].change(Tree(1, 3))
# board[(-1, 1, 0)].change(Tree(1, 2))
# board[(-2, 0, 2)].change(Tree(1, 3))
# board.where(Tree(1, 3))
# board.where(None)
# print(board)
# board.where(1, belongs_to_player)


class PlayerBoard:
    def __init__(self, available=(2, 4, 1, 0), on_player_board=(4, 4, 3, 2), money=2, score=0):
        # self.available = {0: 2, 1: 2, 2: 1, 3: 0}
        self.available = list(available)
        # self.costs = {0: [2, 2, 1, 1], 1: [3, 3, 2, 2], 2: [4, 3, 3], 3: [5, 4]}
        self.costs = [[2, 2, 1, 1], [3, 3, 2, 2], [4, 3, 3], [5, 4]]
        # self.on_player_board = {0: 4, 1: 4, 2: 3, 3: 2}
        self.on_player_board = list(on_player_board)
        self.money = money
        self.score = score

    def __deepcopy__(self, memodict={}):
        new = PlayerBoard(available=self.available, on_player_board=self.on_player_board, money=self.money, score=self.score)
        return new

    def costs_for_each(self):
        return [self.costs[size][self.on_player_board[size] - 1] for size in range(3 + 1)]

    def place(self, size):
        if self.available[size] == 0:
            raise ValueError("none available to place")
        if size > self.money:
            raise ValueError('not enough money')
        self.available[size] -= 1
        self.money -= size

    def purchase(self, size):
        if self.on_player_board[size] == 0:
            raise ValueError("none available to purchase")
        cost = self.costs[size][self.on_player_board[size] - 1]
        if cost > self.money:
            raise ValueError('not enough money')
        self.on_player_board[size] -= 1
        self.available[size] += 1
        self.money -= cost
        return cost

    def return_to_board(self, size):
        if self.on_player_board[size] == len(self.costs[size]):
            return
        self.on_player_board[size] += 1

    def __str__(self):
        output = ''
        output += f'Available Trees: {self.available}\n'
        output += f'On Player Board: {self.on_player_board}\n'
        output += f'Costs For Each Tree: {self.costs_for_each()}\n'
        output += f'Available Money: {self.money}\n'
        output += f'Victory Points: {self.score}\n'
        return output


class Move:
    def __init__(self, move_name, player, coordinates=None, spawn_coordinates=None, size=None):
        self.move_name = move_name
        self.player = player
        self.coordinates = coordinates
        self.spawn_coordinates = spawn_coordinates
        self.size = size

    def __hash__(self):
        return hash(self.move_name) + hash(self.player) + hash(self.coordinates) + hash(self.spawn_coordinates) + hash(self.size)

    def __eq__(self, other):
        return self.move_name == other.move_name and self.player == other.player and self.coordinates == other.coordinates and self.spawn_coordinates == other.spawn_coordinates and self.size == other.size

    def vector(self):
        move_number = move_names.index(self.move_name)
        output_vector = one_hot(move_number, len(move_names))
        if self.coordinates is None:
            output_vector = np.concatenate((output_vector, [0, 0, 0]))
        else:
            output_vector = np.concatenate((output_vector, self.coordinates))
        if self.spawn_coordinates is None:
            output_vector = np.concatenate((output_vector, [0, 0, 0]))
        else:
            output_vector = np.concatenate((output_vector, self.spawn_coordinates))
        if self.size is None:
            output_vector = np.concatenate((output_vector, [0]))
        else:
            output_vector = np.concatenate((output_vector, [self.size]))
        return output_vector

    def __repr__(self):
        if self.move_name == 'plant':
            return f'{self.move_name} {str(self.coordinates)} {self.spawn_coordinates}'
        elif self.move_name == 'special_plant':
            return f'{self.move_name} {str(self.coordinates)}'
        elif self.move_name == 'pass':
            return f'{self.move_name}'
        elif self.move_name == 'upgrade':
            return f'{self.move_name} {str(self.coordinates)}'
        elif self.move_name == 'collect':
            return f'{self.move_name} {str(self.coordinates)}'
        elif self.move_name == 'purchase':
            return f'{self.move_name} {str(self.size)}'
        else:
            return ''


class State:
    def __init__(self):
        self.check_moves_mode = True
        self.board = generate_radial_hex_array(3)
        self.currently_blocked = [generate_radial_hex_array(3, False),
                                  generate_radial_hex_array(3, False),
                                  generate_radial_hex_array(3, False)]
        self.num_players = 2
        self.board_radius = 3
        # self.money = np.full(self.num_players, 2)
        self.vp_stacks = [[20, 21, 22], [16, 16, 17, 17, 18, 18, 19], [13, 13, 14, 14, 16, 16, 17],
                          [12, 12, 12, 13, 13, 13, 14, 14]]  # check
        if self.num_players == 2:
            self.vp_stacks[0] = []
        self.revolution = -1
        self.player_to_move = 0
        self.first_player = 0
        self.sun_position = -1
        self.players_passed = []
        self.preliminary_moves = True
        self.preliminary_move_counter = 0
        self.final_position = False
        self.total_revolutions = 4
        self.block_planting_upgrading = True
        self.winners = []
        self.spaces_accessed = []
        self.player_boards = []
        self.player_has_tree = []
        for ii in range(self.num_players):
            self.player_has_tree.append(generate_radial_hex_array(3, False))
            self.spaces_accessed.append(generate_radial_hex_array(3, False))
            self.player_boards.append(PlayerBoard())

    def __deepcopy__(self, memodict={}):
        new = State()
        new.check_moves_mode = self.check_moves_mode
        new.board = deepcopy(self.board)
        new.currently_blocked = [deepcopy(item) for item in self.currently_blocked]
        new.num_players = self.num_players
        new.board_radius = self.board_radius
        new.vp_stacks = deepcopy(self.vp_stacks)  # check
        new.revolution = self.revolution
        new.player_to_move = self.player_to_move
        new.first_player = self.first_player
        new.sun_position = self.sun_position
        new.players_passed = deepcopy(self.players_passed)
        new.preliminary_moves = self.preliminary_moves
        new.preliminary_move_counter = self.preliminary_move_counter
        new.final_position = self.final_position
        new.block_planting_upgrading = self.block_planting_upgrading
        new.winners = deepcopy(self.winners)
        new.player_has_tree = [deepcopy(item) for item in self.player_has_tree]
        new.spaces_accessed = [deepcopy(item) for item in self.spaces_accessed]
        new.player_boards = [deepcopy(item) for item in self.player_boards]
        return new

    def __eq__(self, other):
        return self.revolution == other.revolution and self.sun_position == other.sun_position and self.player_to_move == other.player_to_move and self.board == other.board and self.vp_stacks == other.vp_stacks

    def final_evaluate(self, player_index):
        if player_index in self.winners:
            if len(self.winners) == 1:
                y = 100
            else:
                y = 0
        else:
            y = -100
        return y


    def update_currently_blocked(self, height, coordinates):
        for displacement in range(1, height + 1):
            blocked_coordinates = translation(coordinates, direction=self.sun_position, distance=displacement)
            try:
                self.currently_blocked[height - 1][blocked_coordinates].change(True)

            except: KeyError

        # if move.move_name == 'upgrade':
        #     pass

    def update_currently_blocked_from_scratch(self):
        self.currently_blocked = [generate_radial_hex_array(self.board_radius, False),
                                  generate_radial_hex_array(self.board_radius, False),
                                  generate_radial_hex_array(self.board_radius, False)]
        for coordinates, element in self.board.element_dict.items():
            if element.obj is not None:
                self.update_currently_blocked(element.obj.height, coordinates)

    def produce_money(self):
        for coordinate, element in self.board.element_dict.items():
            if element.obj is not None:
                height = element.obj.height
                player = element.obj.player
                if height > 0 and not self.currently_blocked[height - 1][coordinate].obj:
                    self.player_boards[player].money += element.obj.height

    def check_plant(self, player, spawn_coordinates, target_coordinates, for_execution=None):

        if for_execution is None:
            for_execution = False
        if self.board[spawn_coordinates].obj is None or self.board[spawn_coordinates].obj.player != player:
            if for_execution:
                raise ValueError('cannot plant from there, no appropriate tree to spawn from')
            else:
                return False
        if distance(spawn_coordinates, target_coordinates) > self.board[spawn_coordinates].obj.height:
            if for_execution:
                raise ValueError('cannot plant there, it is too far from spawning tree')
            else:
                return False
        if self.board[target_coordinates].obj is not None:
            if for_execution:
                raise ValueError('cannot plant there - it is occupied')
            else:
                return False
        if self.spaces_accessed[player][target_coordinates].obj:
            if for_execution:
                raise ValueError('cannot plant there, space already used this turn')
            else:
                return False
        if self.spaces_accessed[player][spawn_coordinates].obj:
            if for_execution:
                raise ValueError('cannot plant from there, space already used this turn')
            else:
                return False
        if self.block_planting_upgrading and self.currently_blocked[0][target_coordinates].obj:
            if for_execution:
                raise ValueError('cannot plant there - it is blocked')
            else:
                return False
        if self.player_boards[player].available[0] == 0:
            if for_execution:
                raise ValueError('cannot plant there - no seeds available')
            else:
                return False
        return True

    def plant(self, player, spawn_coordinates, target_coordinates):
        self.check_plant(player, spawn_coordinates, target_coordinates, for_execution=True)
        self.board[target_coordinates].change(Tree(player, 0))
        self.player_boards[player].place(0)
        self.spaces_accessed[player][target_coordinates].change(True)
        self.spaces_accessed[player][spawn_coordinates].change(True)
        self.player_has_tree[player][target_coordinates].change(True)

    def check_special_plant(self, player, coordinates, for_execution=None):
        if for_execution is None:
            for_execution = False
        if self.board[coordinates].obj is not None:
            if for_execution:
                raise ValueError('cannot plant there - it is occupied')
            else:
                return False
        if self.board[coordinates].get_radius() != self.board_radius:
            if for_execution:
                raise ValueError('cannot plant there - it is not in the outer ring')
            else:
                return False
        return True

    def special_plant(self, player, coordinates):
        self.check_special_plant(player, coordinates, for_execution=True)
        self.board[coordinates].change(Tree(player, 1))
        self.player_boards[player].place(1)
        self.player_has_tree[player][coordinates].change(True)

    def check_upgrade(self, player, coordinates, for_execution=None):
        if for_execution is None:
            for_execution = False
        if self.board[coordinates].obj is None or self.board[coordinates].obj.player != player:
            if for_execution:
                raise ValueError('cannot upgrade there, no appropriate tree to upgrade')
            else:
                return False
        # if self.board[coordinates].obj.height == 2:
        #     pass
        if self.board[coordinates].obj.height == 3:
            if for_execution:
                raise ValueError('cannot upgrade there - tree is already full size')
            else:
                return False
        if self.block_planting_upgrading and self.currently_blocked[self.board[coordinates].obj.height][coordinates].obj:
            if for_execution:
                raise ValueError('cannot upgrade there - it is blocked')
            else:
                return False
        if self.player_boards[player].available[self.board[coordinates].obj.height + 1] == 0:
            if for_execution:
                raise ValueError('cannot upgrade there - no appropriate tree available')
            else:
                return False
        if self.spaces_accessed[player][coordinates].obj:
            if for_execution:
                raise ValueError('cannot plant there, space already used this turn')
            else:
                return False
        if self.player_boards[player].money < self.board[coordinates].obj.height + 1:
            if for_execution:
                raise ValueError('cannot upgrade, not enough money')
            else:
                return False
        return True

    def upgrade(self, player, coordinates):
        self.check_upgrade(player, coordinates, for_execution=True)
        self.board[coordinates].change(Tree(player, self.board[coordinates].obj.height + 1))
        self.player_boards[player].place(self.board[coordinates].obj.height)
        self.player_boards[player].return_to_board(self.board[coordinates].obj.height - 1)
        self.spaces_accessed[player][coordinates].change(True)
        self.update_currently_blocked(self.board[coordinates].obj.height, coordinates)

    def check_collect(self, player, coordinates, for_execution=None):
        if for_execution is None:
            for_execution = False
        if self.board[coordinates].obj is None or self.board[coordinates].obj.player != player or self.board[coordinates].obj.height != 3:
            if for_execution:
                raise ValueError('cannot collect there, no appropriate tree to collect')
            else:
                return False
        if self.spaces_accessed[player][coordinates].obj:
            if for_execution:
                raise ValueError('cannot collect there, space already used this turn')
            else:
                return False
        if self.player_boards[player].money < 4:
            if for_execution:
                raise ValueError('cannot collect, not enough money')
            else:
                return False
        return True

    def collect(self, player, coordinates):
        self.check_collect(player, coordinates, for_execution=True)
        radius = self.board[coordinates].get_radius()
        while len(self.vp_stacks[radius]) == 0:
            radius += 1
            if radius == 3:
                break
        score = self.vp_stacks[radius].pop()
        self.player_boards[player].score += score
        self.board[coordinates].change(None)
        self.spaces_accessed[player][coordinates].change(True)
        self.update_currently_blocked_from_scratch()
        self.player_has_tree[player][coordinates].change(False)
        # print(f'player: {player} scored {score}')

    def check_purchase(self, player, size, for_execution=None):
        if for_execution is None:
            for_execution = False
        if self.player_boards[player].on_player_board[size] == 0:
            if for_execution:
                raise ValueError('cannot purchase, no appropriate tree on player board')
            else:
                return False
        if self.player_boards[player].costs[size][self.player_boards[player].on_player_board[size] - 1] > self.player_boards[player].money:
            if for_execution:
                raise ValueError('not enough money')
            else:
                return False

        return True

    def purchase(self, player, size):
        self.check_purchase(player, size, for_execution=True)
        self.player_boards[player].money -= self.player_boards[player].costs[size][self.player_boards[player].on_player_board[size] - 1]
        self.player_boards[player].on_player_board[size] -= 1
        self.player_boards[player].available[size] += 1

    def next_move(self):
        if self.preliminary_moves:
            self.preliminary_move_counter += 1
            if self.preliminary_move_counter < self.num_players:
                self.player_to_move = (self.player_to_move + 1) % self.num_players
            elif self.preliminary_move_counter == self.num_players:
                pass
            elif self.preliminary_move_counter < self.num_players * 2:
                self.player_to_move = (self.player_to_move - 1) % self.num_players
            elif self.preliminary_move_counter == self.num_players * 2:
                self.next_round()
        else:
            if len(self.players_passed) == self.num_players:
                self.next_round()
            else:
                self.player_to_move = (self.player_to_move + 1) % self.num_players
                while self.player_to_move in self.players_passed:
                    self.player_to_move = (self.player_to_move + 1) % self.num_players

    def next_round(self):
        self.players_passed = []
        self.spaces_accessed = [generate_radial_hex_array(3, False),
                                generate_radial_hex_array(3, False),
                                generate_radial_hex_array(3, False)]
        if not self.preliminary_moves:
            self.first_player = (self.first_player + 1) % self.num_players
            self.player_to_move = self.first_player
        else:
            self.player_to_move = self.first_player = 0
        self.preliminary_moves = False
        self.preliminary_move_counter = 0
        self.preliminary_moves = False
        self.sun_position = (self.sun_position + 1) % 6
        if self.sun_position == 0:
            self.revolution = self.revolution + 1
        if self.revolution == self.total_revolutions:
            self.final_position = True
            return
        self.update_currently_blocked_from_scratch()
        self.produce_money()

    def pass_player(self, player):
        self.players_passed.append(player)
        self.next_move()

    def execute_move(self, move):
        if move.player != self.player_to_move:
            return
        if move.move_name == 'plant':
            self.plant(move.player, move.spawn_coordinates, move.coordinates)
        elif move.move_name == 'special_plant':
            self.special_plant(move.player, move.coordinates)
        elif move.move_name == 'upgrade':
            self.upgrade(move.player, move.coordinates)
        elif move.move_name == 'collect':
            self.collect(move.player, move.coordinates)
        elif move.move_name == 'purchase':
            self.purchase(move.player, move.size)
        elif move.move_name == 'pass':
            self.pass_player(move.player)
        else:
            raise TypeError('invalid move name')
        self.next_move()

    def check_move(self, move, for_execution=None):
        if for_execution is None:
            for_execution = False
        if move.player != self.player_to_move:
            return False
        if move.move_name == 'plant':
            return self.check_plant(move.player, move.spawn_coordinates, move.coordinates, for_execution=for_execution)
        elif move.move_name == 'special_plant':
            return self.check_special_plant(move.player, move.coordinates, for_execution=for_execution)
        elif move.move_name == 'pass':
            return not self.preliminary_moves
        elif move.move_name == 'upgrade':
            return self.check_upgrade(move.player, move.coordinates, for_execution=for_execution)
        elif move.move_name == 'collect':
            return self.check_collect(move.player, move.coordinates, for_execution=for_execution)
        elif move.move_name == 'purchase':
            return self.check_purchase(move.player, move.size, for_execution=for_execution)
        else:
            return False

    def generate_move_list(self, player=None):
        if player is None:
            player = self.player_to_move
        has_tree = self.player_has_tree[player].where(True)
        no_trees = self.board.where(None)
        can_use = self.spaces_accessed[player].where(False)
        has_usable_tree = [ii for ii in has_tree if ii in can_use]
        usable_empty_space = [ii for ii in no_trees if ii in can_use]
        move_list = []
        if self.preliminary_moves:
            move_name = 'special_plant'
            for coordinates in no_trees:
                move = Move(move_name, player, coordinates=coordinates)
                if not self.check_moves_mode or self.check_move(move):
                    move_list.append(move)
        else:
            move_name = 'upgrade'
            for coordinates in has_usable_tree:
                move = Move(move_name, player, coordinates=coordinates)
                if not self.check_moves_mode or self.check_move(move):
                    move_list.append(move)
            move_name = 'collect'
            for coordinates in has_usable_tree:
                move = Move(move_name, player, coordinates=coordinates)
                if not self.check_moves_mode or self.check_move(move):
                    move_list.append(move)
            move_name = 'plant'
            for spawn_coordinates in has_usable_tree:
                # check if yours to make faster
                for target_coordinates in usable_empty_space:
                    move = Move(move_name, player, coordinates=target_coordinates, spawn_coordinates=spawn_coordinates)
                    if not self.check_moves_mode or self.check_move(move):
                        move_list.append(move)
            move_name = 'purchase'
            for size in range(3 + 1):
                move = Move(move_name, player, size=size)
                if not self.check_moves_mode or self.check_move(move):
                    move_list.append(move)
            move_name = 'pass'
            move = Move(move_name, player)
            move_list.append(move)
        return move_list

    def generate_featurespace(self, player):
        """
        for two player only
        :param player:
        :return:
        """
        output = []
        to_move = 1 if self.player_to_move == player else 0
        output.append(to_move)
        score_normalizer = 75
        score = self.player_boards[player].score
        output.append(score / score_normalizer)
        opponent_scores = [self.player_boards[plr].score / score_normalizer for plr in range(self.num_players) if plr != player]
        output.extend(opponent_scores)
        money_normalizer = 10
        money = self.player_boards[player].money
        output.append(money / money_normalizer)
        opponent_moneys = [self.player_boards[plr].money / money_normalizer for plr in range(self.num_players) if plr != player]
        output.extend(opponent_moneys)
        output.append(self.revolution/self.total_revolutions)
        output.append(self.sun_position - 2)
        available = self.player_boards[player].available
        output.extend(available)
        opponent_available = [self.player_boards[plr].available for plr in range(self.num_players) if plr != player]
        for opponent in opponent_available:
            output.extend(opponent)
        on_player_board = self.player_boards[player].on_player_board
        output.extend(on_player_board)
        opponent_on_player_board = [self.player_boards[plr].on_player_board for plr in range(self.num_players) if plr != player]
        for opponent in opponent_on_player_board:
            output.extend(opponent)
        tree_size_normalizer = 4

        tile_features = [0] * (3 * self.board_radius * (self.board_radius + 1) + 1)
        has_tree = self.player_has_tree[player].where(True)
        for coordinates in has_tree:
            layers = [0, 5, 11, 18, 25, 31, 36]
            index = layers[coordinates[0] + 3] + coordinates[1] + max(0, (-3 - coordinates[0]))
            tile_features[index] = (self.board[coordinates].obj.height + 1) / tree_size_normalizer
        output.extend(tile_features)

        for plr in range(self.num_players):
            if plr != player:
                opp_tile_features = [0] * (3 * self.board_radius * (self.board_radius + 1) + 1)
                opp_has_tree = self.player_has_tree[player].where(True)
                for coordinates in opp_has_tree:
                    layers = [0, 5, 11, 18, 25, 31, 36]
                    index = layers[coordinates[0] + 3] + coordinates[1] + max(0, (-3 - coordinates[0]))
                    opp_tile_features[index] = (self.board[coordinates].obj.height + 1) / tree_size_normalizer
                output.extend(opp_tile_features)

        return np.array([output])

    def __str__(self):
        output = ''
        if self.revolution >= 0:
            output += f'Revolution: {self.revolution}\n'
            output += f'Sun Position: {self.sun_position}\n'
        else:
            output += f'Revolution: preliminary phase\n'
        for player in range(self.num_players):
            output += f'Player: {player}: \n {str(self.player_boards[player])} \n'
        output += str(self.board)
        return output
#
# state = State()
# State.attribute_list = [a for a in dir(state) if not a.startswith('__')]


class LearningNode:
    def __init__(self, state, player_point_of_view=None):
        self.player_point_of_view = player_point_of_view
        if player_point_of_view is None:
            self.player_point_of_view = state.player_to_move
        self.state = state
        self.children = {}
        self.raw_value = None
        self.value = None
        self.best_move = None
        self.expanded = False

    def __getitem__(self, move):
        return self.children[move]

    def generate_children(self):
        if self.state.final_position:
            return
        move_list = self.state.generate_move_list(self.state.player_to_move)
        for ii, move in enumerate(move_list):
            new_state = deepcopy(self.state)
            new_state.execute_move(move)
            self.children[move] = LearningNode(new_state, player_point_of_view=self.player_point_of_view)

    def evaluate(self, learning_object):
        """
        sets the raw_value of the node parameter based on either the terminal state or the learning object
        :param learning_object:
        :return:
        """
        if self.state.final_position:
            self.raw_value = self.state.final_evaluate(self.player_point_of_view)
        else:
            X = self.state.generate_featurespace(self.player_point_of_view)
            self.raw_value = learning_object.predict(X)[0]

    def minimax_evaluate(self):
        """
        sets the 'true' value of the node to the minimax evaluation of all children (uses raw_value if no known children
        are searched for
        :return:
        """
        if len(self.children) == 0:
            self.value = self.raw_value
            return
        value = None
        for move, child in self.children.items():
            # if not child.expanded:
            #     child.generate_children()
            child.minimax_evaluate()
            if value is None:
                value = child.value
            if self.player_point_of_view == child.state.player_to_move: # evaluate the best case for player
                if child.value > value:
                    value = child.value
            else: # evaluate the worst case for player since the other player gets to move
                if child.value < value:
                    value = child.value
        self.value = value

    def expand_to_depth(self, learning_object, max_depth):
        # print(max_depth)
        # print(self.state.board)
        if self.raw_value is None:
            self.evaluate(learning_object)
        if max_depth > 0:
            if not self.expanded:
                self.generate_children()
                self.expanded = True
        # if max_depth > 0:
            for move, child in self.children.items():
                child.expand_to_depth(learning_object, max_depth - 1) ## infinite recursion error here?
        self.minimax_evaluate()


class Player:
    def __init__(self, policy, learning=False, starting_learning_object=None, learning_tree=None):
        self.policy = policy
        self.learning_object = starting_learning_object
        self.learning = learning
        self.learning_iterations = 0
        self.learning_tree = learning_tree

    def choose_move(self, state, max_depth):
        if self.learning:
            move, learning_tree = self.policy(state, learning_object=self.learning_object, learning_tree=self.learning_tree, max_depth=max_depth)
            self.learning_tree = learning_tree
            return move
        else:
            return self.policy(state)

    def learn(self, X, y):
        if self.learning:
            self.learning_object.partial_fit(X, y)
            self.learning_iterations += len(X)


class GameSettings:
    def __init__(self, max_depth):
        self.max_depth = max_depth


class Game:
    def __init__(self, player_list, game_settings, starting_state=None):
        if starting_state is None:
            starting_state = State()
        self.player_list = player_list
        self.game_settings = game_settings
        self.state = starting_state
        self.has_learning_player = any([player.learning for player in player_list])
        self.X = {ii: None for ii, player in enumerate(player_list) if player.learning}

    def play(self):
        player_index = self.state.player_to_move
        player = self.player_list[player_index]
        move = player.choose_move(self.state, max_depth=self.game_settings.max_depth)
        if self.has_learning_player and player.learning:
            if self.X[player_index] is None:
                self.X[player_index] = self.state.generate_featurespace(player_index)
            else:
                self.X[player_index] = np.append(self.X[player_index], self.state.generate_featurespace(player_index), axis=0)
            if self.game_settings.max_depth > 0:
                player.learning_tree = player.learning_tree[move]
        self.state.execute_move(move)

    def play_through(self):
        while True:
            self.play()
            if self.state.final_position:
                final_score = np.array([self.state.player_boards[player_index].score for player_index in range(len(self.player_list))])
                winning_score = max(final_score)
                for player_index, score in enumerate(final_score):
                    if score == winning_score:
                        self.state.winners.append(player_index)
                for player_index, player in enumerate(self.player_list):
                    if player.learning:
                        X = self.X[player_index]
                        y = np.ones(len(X)) * self.state.final_evaluate(player_index)

                        # if final_score[player_index] == max(final_score):
                        #     y = np.ones(len(X)) * 100
                        # elif final_score[player_index] == min(final_score):
                        #     y = np.ones(len(X)) * -100
                        # else:
                        #     y = np.ones(len(X)) * 0
                        # if np.all(final_score == final_score[0]):
                        #     y = np.ones(len(X)) * 0
                        player.learn(X, y)
                break
        return final_score

    def __str__(self):
        return str(self.state)

