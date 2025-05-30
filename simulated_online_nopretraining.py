from MDP_functions import get_data, find_next_state, transition_probabilities_faster
from utils import *
from MDP_generator import *
import os
import time
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle

np.random.seed(0)

dataset = 'simulation'
num_episodes_online = 30000
state_abstraction = False
k = None
benchmark = False
learning_rate = 0.8
discount_factor = 0.9
epsilon_start = 0.1
epsilon_end = 0.01
decay_rate = np.log(epsilon_end / epsilon_start) / num_episodes_online
epsilon = epsilon_start

print("=========Loading Environment==========")
env = InterpretableLoanMDP()
n_actions = env.action_space.n
Q_table = np.zeros((0, n_actions))
state_to_index = {}

def get_state_index(state, Q_table, state_to_index):
    state_tuple = tuple(state)
    if state_tuple in state_to_index:
        return state_to_index[state_tuple], Q_table

    #new state encountered
    index = len(Q_table)
    state_to_index[state_tuple] = index
    new_row = np.zeros((1, Q_table.shape[1]))
    Q_table = np.vstack([Q_table, new_row])
    return index, Q_table

print(f"=====Online learning for {dataset}, state abstraction: {state_abstraction}, k: {k}====")

cumulative_reward = []
list_len_ep = []
list_granted = []
optimal_paths = []
n_granted = 0
incorrect_transitions = 0
total_visited_states = 0
real_unseen_states = 0
n_unseen_cases = 0
n_unseen_granted_cases = 0
start_time = time.time()

#Online Q-learning loop
with tqdm(range(num_episodes_online), desc="Online Training Progress", unit="episode") as pbar:
    for episode in pbar:
        state = env.reset()
        reward_episode = 0
        len_epis = 0
        done = False
        granted = False
        episode_path = []

        state_id, Q_table = get_state_index(state, Q_table, state_to_index)

        while not done and len_epis < 100:
            possible_actions = env.get_valid_actions()

            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)  # Exploration
            else:
                action = np.argmax(Q_table[state_id])  # Exploitation

            if action not in possible_actions:
                incorrect_transitions += 1

            next_state, reward, done, granted, info = env.step(action)
            next_state_id, Q_table = get_state_index(next_state, Q_table, state_to_index)

            #track unseen state stats
            total_visited_states += 1
            if next_state_id >= len(Q_table) - 1:  
                real_unseen_states += 1

            #bellman update
            Q_table[state_id, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state_id]) - Q_table[state_id, action])

            state = next_state
            state_id = next_state_id
            reward_episode += reward
            len_epis += 1
            episode_path.append(state)

        if granted:
            n_granted += 1

        cumulative_reward.append(reward_episode)
        list_len_ep.append(len_epis)
        list_granted.append(100 * (n_granted / (episode + 1)))
        episode_path.append(reward_episode)
        optimal_paths.append(episode_path)

        #epsilon decay
        epsilon = max(epsilon_end, epsilon_start * np.exp(decay_rate * episode))

        pbar.set_postfix(success=f"{100 * (n_granted / (episode + 1)):.2f}%", len_episode=f"{int(np.mean(list_len_ep))}")

end_time = time.time()
runtime = end_time - start_time


print("=====Online Learning Results====")
print("Online Training runtime:", runtime)
print(f"Online success rate: {100 * (n_granted / num_episodes_online):.2f}%")
print(f"% of unseen states: {(real_unseen_states / total_visited_states) * 100:.2f}%")
print(f"Average episode length (last 100): {np.mean(list_len_ep[-100:]):.2f}")


os.makedirs("results", exist_ok=True)
training_results = {
    "state_abstraction": state_abstraction,
    "k": k,
    "n_states_online": Q_table.shape[0],
    "unseen_states": real_unseen_states,
    "total_visited_states": total_visited_states,
    "incorrect_transitions": incorrect_transitions,
    "n_unseen_granted_cases": n_unseen_granted_cases,
    "n_unseen_cases": n_unseen_cases,
    "cumulative_reward": cumulative_reward,
    "len_ep": list_len_ep,
    "granted": list_granted,
    "n_granted": n_granted,
    "runtime": runtime,
}

filename = f"results/{dataset}_{state_abstraction}_{benchmark}_online_nopretraining_{k}.pkl"
with open(filename, 'wb') as f:
    pickle.dump(training_results, f)
