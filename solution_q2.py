import gymnasium as gym 

# env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode='human', is_slippery=True, ) #initialization 
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, ) #initialization 
state, info = env.reset()

# estimate T and R
p1, p2, t = dict(), dict(), dict()
r = dict()

for _ in range(1000): 
    action = env.action_space.sample()  # random action 
    new_state, reward, terminated, truncated, info = env.step(action) 
    # print(action, observation, reward, terminated, truncated, info)
 
    # count instances of s,a,s' and store in p1, count instances of s,a and store in p2
    if (state, action, new_state) not in p1:
        p1[(state, action, new_state)] = 0
    if (state, action) not in p2:
        p2[(state, action)] = 0
    p1[(state, action, new_state)] += 1
    p2[(state, action)] += 1

    if terminated or truncated:
        state, info = env.reset()
    else:
        state = new_state

# combine p1 and p2 to estimate t
for k1 in p1:
    for k2 in p2:
        if k1[0]==k2[0] and k1[1]==k2[1]:
            t[k1] = p1[k1]/p2[k2]
            print(f'T{k1} = {t[k1]}')

env.close()