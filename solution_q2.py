import gymnasium as gym 

# env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode='human', is_slippery=True, ) #initialization 
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, ) #initialization 

# Random Policy until 1000 episodes

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
    
    # Assign rewards based on the new state and termination condition
    if terminated:
        if reward > 0: # Goal reached
            r[(state, action, new_state)] = 1
        else: # hole reached
            r[(state, action, new_state)] = 0
    else: # frozen tile
        r[(state, action, new_state)] = 0

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


# Value iteration

import numpy as np

num_states = env.observation_space.n
num_actions = env.action_space.n
V = np.zeros(num_states)  # Value function initialization
gamma = 0.9  # Discount factor
theta = 0.0001  # Convergence threshold
delta = 0

while True:
    delta = 0
    for s in range(num_states):
        v = V[s]
        V[s] = max([sum([t.get((s, a, s_prime), 0) * (r.get((s, a, s_prime), 0) + gamma * V[s_prime]) for s_prime in range(num_states)]) for a in range(num_actions)])
        delta = max(delta, abs(v - V[s]))
    
    if delta < theta:
        break

# V now contains the optimal value function


# Policy Iteration
policy = np.zeros(num_states, dtype=int)

for s in range(num_states):
    action_values = []
    for a in range(num_actions):
        action_value = sum([t.get((s, a, s_prime), 0) * (r.get((s, a, s_prime), 0) + gamma * V[s_prime]) for s_prime in range(num_states)])
        action_values.append(action_value)
    best_action = np.argmax(action_values)
    policy[s] = best_action

# policy now contains the optimal policy



#optimal policy

total_episodes = 1000  # Total number of episodes you want to simulate
total_rewards = 0  # To keep track of the total rewards
unique_transitions = {}  # Dictionary to track unique transitions

print("-" * 100)
print(" Now using the Optimal policy\n")

for _ in range(total_episodes):
    state, info = env.reset()  # Reset environment for a new episode
    done = False

    while not done:
        action = policy[state]  # Choose action based on the optimal policy
        new_state, reward, done, truncated, info = env.step(action)  # Take the action

        # Check if the transition is unique and print it
        transition_key = (state, action, new_state)
        if transition_key not in unique_transitions:
            unique_transitions[transition_key] = True
            print(f"T({state}, {action}, {new_state}) = {t.get(transition_key, 0)}")

        total_rewards += reward  # Update the rewards
        state = new_state  # Update the state

env.close()




