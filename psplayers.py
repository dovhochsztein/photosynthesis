from photosynth import *
import sklearn
from sklearn import neural_network as nn
import numpy as np
import scipy
import copy


def random_policy_generator(given_distribution_function=None):
    def random_policy(state):
        move_list = state.generate_move_list()
        if given_distribution_function is None:
            return random.choice(move_list)
        distribution = given_distribution_function(move_list)
        return np.random.choice(move_list, p=distribution)
    return random_policy



def distribution_function(move_list):
    distribution = np.zeros(len(move_list))
    likelihood_dict = {'plant': 2, 'special_plant': 100, 'pass': .3, 'upgrade': 8, 'collect': 1, 'purchase': 4}
    move_number_dict = {'plant': 0, 'special_plant': 0, 'pass': 0, 'upgrade': 0, 'collect': 0, 'purchase': 0}
    for move in move_list:
        move_number_dict[move.move_name] += 1
    for ii, move in enumerate(move_list):
        name = move.move_name
        value = move_number_dict[name] * likelihood_dict[name]
        distribution[ii] = value
    distribution /= np.sum(distribution)
    return distribution




def user_input_policy(state):
    move_list = state.generate_move_list()
    for ii, move in enumerate(move_list):
        print(f'Move Choice {ii}: {str(move)}')
    choice = (input('select a move by typing in the move number:\n'))
    while choice == '' or int(choice) not in range(len(move_list)):
        print('invalid. tray again:\n')
        choice = (input('select a move by typing in the move number:\n'))
    return move_list[int(choice)]


def human_player():
    return Player(user_input_policy)


def random_player():
    return Player(random_policy_generator())


def improved_random_player():
    return Player(random_policy_generator(given_distribution_function=distribution_function))


def learning_player(learning_policy, starting_learning_object):
    return Player(learning_policy, learning=True, starting_learning_object=starting_learning_object)


def starting_learning_object_1():
    state = State()
    learning_object = nn.MLPRegressor()
    X = state.generate_featurespace(state.player_to_move)
    y = [0]
    learning_object.fit(X, y)
    return learning_object

def starting_learning_object_2():
    state = State()
    # learning_tree = LearningNode(state)
    learning_object = nn.MLPRegressor()
    X = state.generate_featurespace(state.player_to_move)
    y = [0]
    learning_object.fit(X, y)
    return learning_object#, learning_tree

# import time
def learning_policy_1(state, learning_object, learning_tree=None, max_depth=2):
    move_list = state.generate_move_list()
    values = np.zeros(len(move_list))
    for ii, move in enumerate(move_list):
        new_state = copy.deepcopy(state)
        X = new_state.generate_featurespace(state.player_to_move)
        values[ii] = learning_object.predict(X)
    distribution = scipy.special.softmax(values)
    return np.random.choice(move_list, p=distribution), None

def learning_policy_2(state, learning_object, learning_tree, max_depth=2):
    if max_depth > 0:
        if learning_tree is None or learning_tree.state != state:
            learning_tree = LearningNode(state)
        learning_tree.expand_to_depth(learning_object, max_depth)
        move_list = list(learning_tree.children.keys())
        values = np.zeros(len(move_list))
        for ii, child in enumerate(learning_tree.children.values()):
            values[ii] = child.value
    else:
        move_list = state.generate_move_list()
        values = np.zeros(len(move_list))
        for ii, move in enumerate(move_list):
            new_state = copy.deepcopy(state)
            X = new_state.generate_featurespace(state.player_to_move)
            values[ii] = learning_object.predict(X)
    distribution = scipy.special.softmax(values)
    return np.random.choice(move_list, p=distribution), learning_tree



learning_object = nn.MLPRegressor()
np.random.seed(0)
X = np.random.rand(10, 10)
Y = np.random.rand(10)
learning_object.fit(X,Y)
learning_object.predict(np.random.rand(1,10))