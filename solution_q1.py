import gymnasium as gym

env = gym.make('Blackjack-v1', natural=False, sab=False)

# print(env.action_space)
# print(env.observation_space)

###YOUR Q-LEARNING CODE BEGINS
qtable = dict()
ntable = dict() # track number of visits for exploration function
lr = .5
discount = .5
iters = int(1e5)

# define exploration function
def f(u, n, k=100.0):
    # return u
    return u + k/n

# start first game
state, info = env.reset()

# perform iters-many Q-updates
for i in range(iters):
    
    # initialize Q-table with zeros
    for key in (state, 0), (state, 1):
        if key not in qtable:
            qtable[key] = 0
            ntable[key] = 1
    
    # follow qtable policy
    # action = env.action_space.sample() # random action
    action = 1 if qtable[(state, 1)] > qtable[(state, 0)] else 0
    ntable[(state, action)] += 1
    
    new_state, reward, terminated, truncated, info = env.step(action)
    
    # initialize Q-table with zeros
    for key in (new_state, 0), (new_state, 1):
        if key not in qtable:
            qtable[key] = 0
            ntable[key] = 1

    # make a Q-update
    sample = reward + discount * max(f(qtable[(new_state, 0)], ntable[(new_state, 0)]),
                                     f(qtable[(new_state, 1)], ntable[(new_state, 1)]))
    qtable[(state, action)] = (1 - lr) * qtable[(state, action)] + lr * sample
    # print(observation, action, reward, terminated, truncated, info)

    # start new game
    if terminated or truncated:
        state, info = env.reset()
    # continue current game
    else:
        state = new_state
    
    # decrease learning rate
    lr *= (1 - 1.0*i/iters)

env.close()

print('Final Q-values:')
for key in qtable:
    print(f'Q{key} = {qtable[key]}')


# demonstrate qtable policy
env2 = gym.make("Blackjack-v1", render_mode="human", natural=False, sab=False) # Initializing environments
state, info = env2.reset()

wins = 0
total_games = 0

for _ in range(50):
    # follow qtable policy
    action = 1 if qtable[(state, 1)] > qtable[(state, 0)] else 0
    state, reward, terminated, truncated, info = env.step(action)
    
    # start new game
    if terminated or truncated:
        # print(reward)
        if reward == 1:
            wins += 1
        total_games += 1
        state, info = env2.reset()

env2.close()

print()
print(f'Rate of victory in 50 games: {100.0*wins/total_games}%')
# print(len(qtable.keys()))
###YOUR Q-LEARNING CODE ENDS