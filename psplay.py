import time
from psplayers import *
import cProfile
import pickle
now = time.time()

g = State()
g.generate_featurespace(1)

pr = cProfile.Profile()
pr.enable()

random.seed(2)
# g = State()
# for ii in range(6):
#     g.execute_move(random.choice(g.generate_move_list()))
# print(g)
# move_list = g.generate_move_list()
# M=random.choice(g.generate_move_list())
# g.execute_move(M)
# g.execute_move(random.choice(g.generate_move_list()))


learning_player_1 = learning_player(learning_policy_1, starting_learning_object=starting_learning_object_1())
learning_player_2 = learning_player(learning_policy_1, starting_learning_object=starting_learning_object_1())
learning_player_3 = learning_player(learning_policy_2, starting_learning_object=starting_learning_object_2())
learning_player_4 = learning_player(learning_policy_2, starting_learning_object=starting_learning_object_2())

# with open('player1.player', 'rb') as file:
#     learning_player_1 = pickle.load(file)
# with open('player2.player', 'rb') as file:
#     learning_player_2 = pickle.load(file)

with open('player10.player', 'rb') as file:
    learning_player_10 = pickle.load(file)
with open('player20.player', 'rb') as file:
    learning_player_20 = pickle.load(file)



N = 20
games = [None] * N
won = 0
drawn = 0
lost = 0
for ii in range(N):
    # games[ii] = Game([learning_player_3, learning_player_4], GameSettings(2))
    game = Game([learning_player_3, learning_player_4], GameSettings(2))
    # final_score = games[ii].play_through()
    final_score = game.play_through()
    if final_score[0] > max(final_score[1:]):
        won += 1
    elif final_score[0] == max(final_score[1:]):
        drawn += 1
    else:
        lost += 1
    print(f'Game: {ii}; Score: {final_score}; Cumulative_Time: {time.time() - now}')
    # print(games[ii])
# H = Game([human_player(), improved_random_player(), improved_random_player()])
# for ii in range(1000):
#     print(f'move: {ii}')
#     G.play()
#     if G.state.final_position:
#         break
# print(G)
# H.play_through()
# print(H)
print(f'W: {won}, L: {lost}, D: {drawn}')
time_per_game = (time.time() - now)/N
print(time_per_game)

with open('player3.player', 'wb') as file:
    pickle.dump(learning_player_3, file)
with open('player4.player', 'wb') as file:
    pickle.dump(learning_player_4, file)


pr.print_stats(sort='cumtime')
