import gymnasium as gym

env = gym.make('Blackjack-v1', natural=False, sab=False)

# print(env.action_space)
# print(env.observation_space)

###YOUR Q-LEARNING CODE BEGINS
qtable = dict()
lr = .5
discount = .5
n = int(1e5)

# start first game
state, info = env.reset()

# perform Q-learning on n samples
for i in range(n):
    action = env.action_space.sample() # agent policy that uses the observation and info
    new_state, reward, terminated, truncated, info = env.step(action)
    
    # initialize Q-table with zeros
    for key in (state, action), (new_state, 0), (new_state, 1):
        if key not in qtable:
            qtable[key] = 0

    # make a Q-update
    sample = reward + discount * max(qtable[(new_state, 0)], qtable[(new_state, 1)])
    qtable[(state, action)] = (1 - lr) * qtable[(state, action)] + lr * sample
    # print(observation, action, reward, terminated, truncated, info)

    # start new game
    if terminated or truncated:
        state, info = env.reset()
    # continue current game
    else:
        state = new_state
    
    # decrease learning rate
    lr *= (1 - 1.0*i/n)

env.close()

for key in qtable:
    print(f'Q{key} = {qtable[key]}')
###YOUR Q-LEARNING CODE ENDS