import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import pickle
import time
import math
import os
from sklearn.preprocessing import LabelEncoder
from MDP_functions import get_data, find_next_state, transition_probabilities_faster, define_state_cols, get_reward
np.random.seed(0)


state_abstraction = sys.argv[1]
if state_abstraction == 'False':
    state_abstraction = False
    k = None
elif state_abstraction =='structural':
    k = 1
elif state_abstraction == 'last_action':
    k = None
else:
    k = int(sys.argv[2])

dataset = sys.argv[3]
benchmark = False

def convert_to_tuple(path):
    return tuple(tuple(x) if isinstance(x, list) else x for x in path)


def testing_RL(Q_table, test_case_ids):
    optimal_paths = []
    n_granted_test = 0
    cumulative_reward_test = []
    list_granted_test = []
    list_len_ep = []
    case_id_tracker = []
    n_stopping_cases = 0
    n_stopping_cases_next = 0
    epsilon = 0
    
    
    with tqdm(range(num_episodes_test), desc="Testing Progress", unit="episode") as pbar:
        for episode in pbar:
            reward_episode = 0
            len_epis = 0

            # randomly sampling an event trace
            current_trace_id = np.random.choice(test_case_ids) 
            case_id_tracker.append(current_trace_id)
            current_trace = df[df['ID'] == current_trace_id]
            current_state_unabs = all_state_unabs_index[tuple(current_trace.iloc[0][state_cols_simulation])]
            current_state = all_state_index[tuple(current_trace.iloc[0][state_cols])]
            action = 0
        
            Q_table_previous = Q_table
            episode_path = []
        
            while action not in terminal_actions:
                
                possible_actions = [action for action in range(n_actions)
                                if (current_state_unabs, action) in transition_proba and transition_proba[(current_state_unabs, action)].sum() > 0]
                if len(possible_actions) == 0:
                        Q_table = Q_table_previous
                        n_stopping_cases +=1
                        break
            
                max_value_indices = np.where(Q_table[current_state, possible_actions] == np.max(Q_table[current_state, possible_actions]))[0]
                action = possible_actions[np.random.choice(max_value_indices)]
                
                #move to the next state
                next_state_unabs, proba = find_next_state(current_state_unabs, action, transition_proba, n_states_unabs)
         
                #get the abstracted version of the next state
        
                next_state = unabs_to_abs_state.get(tuple(all_states_unabs.iloc[next_state_unabs]), None)
                
                if len_epis >= stopping_criteria or next_state is None: #reverse the episode
                    Q_table = Q_table_previous
                    n_stopping_cases_next +=1
                    break
            
                #define reward
                reward = get_reward(dataset, action, budget, cost, activity_index)
                
                episode_path.append(current_state)
                episode_path.append([key for key, value in activity_index.items() if value == action])

                #update Q-value using the Q-learning update rule
                Q_table[current_state, action] += learning_rate * (reward + discount_factor *np.max(Q_table[next_state]) - Q_table[current_state, action])
                
                current_state = next_state  
                current_state_unabs = next_state_unabs
                reward_episode += reward
                len_epis += 1        
            
            if dataset == 'bpi2017' and action == activity_index['A_Pending']:
                n_granted_test +=1
        
            elif dataset == 'bpi2012' and action == activity_index['A_APPROVED']:
                n_granted_test +=1
           

            cumulative_reward_test.append(reward_episode)
            list_granted_test.append(100*(n_granted_test/(episode+1)))
            list_len_ep.append(len_epis)
            episode_path.append(current_state)
            episode_path.append(reward_episode)
            optimal_paths.append(episode_path)

            
            epsilon = max(epsilon_end, epsilon - epsilon_decay) # decay epsilon
            pbar.set_postfix(success=f"{100 * (n_granted_test / (episode + 1)):.2f}%")
        
    end_time = time.time()
    runtime = end_time - start_time  
    test_df = df[df['ID'].isin(test_case_ids)]
    uplift = ((n_granted_test/num_episodes_test)-(len(test_df[test_df["Outcome"]=='Success']['ID'].unique())/len(test_case_ids))) *100 
    print(f"uplift: {uplift}%")

    # count frequencies of each unique path
    hashable_optimal_paths = [convert_to_tuple(path) for path in optimal_paths]
    optimal_path_frequencies = Counter(hashable_optimal_paths)
    sorted_paths = sorted(optimal_path_frequencies.items(), key=lambda x: x[1], reverse=True)

    results = {"state_abstraction": state_abstraction,
               "k": k,
               "n_states": n_states,
               "case_id": case_id_tracker,
               "Q_table": Q_table,
                                "average uplift": uplift,
                                "optimal_paths": optimal_paths,
                                "optimal_path_frequencies": optimal_path_frequencies,
                                "cumulative_reward": cumulative_reward_test,
                                "len_ep": list_len_ep,
                                "granted": list_granted_test,
                                "n_granted": n_granted_test,
                                "runtime": runtime,
                                "budget": budget}
    print(np.mean(list_len_ep))
    return results



df, df_success, all_cases, all_actions, activity_index, n_actions, budget = get_data(dataset, k, state_abstraction)
shuffled_cases = np.random.permutation(all_cases)
#split into train and test set
train_size = int(0.8 * len(all_cases))
train_case_ids = shuffled_cases[:train_size] 
test_case_ids = shuffled_cases[train_size:] 

if dataset == 'bpi2017':
    num_times = 10
elif dataset == 'bpi2012':
    num_times = 40

#define the hyperparameters 
num_episodes = int(len(train_case_ids) * num_times) 
num_episodes_test = len(test_case_ids)
learning_rate = 0.8
discount_factor = 0.9
stopping_criteria = 100 #max length of traces
cost = 1 #cost of individual action
epsilon_start = 0.1
epsilon_end = 0.01
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes  
epsilon = epsilon_start


print(f"=====Data preparation for {dataset}, state abstraction: {state_abstraction}, k: {k}====")
state_cols, state_cols_simulation, terminal_actions = define_state_cols(dataset, df, k, state_abstraction, benchmark, activity_index)
print(state_cols)
epsilon = epsilon_start
start_time = time.time()
df, df_success, all_cases, all_actions, activity_index, n_actions, budget = get_data(dataset, k, state_abstraction)
all_states_unabs = df[state_cols_simulation].drop_duplicates().reset_index(drop=True)
n_states_unabs = len(all_states_unabs)
print(f"Simulating the environment with {n_states_unabs} distinct states")
transition_proba = transition_probabilities_faster(df, state_cols_simulation, all_states_unabs, activity_index, n_actions)
all_states_unabs = all_states_unabs.drop(columns=["state_index"])
all_states = df[state_cols].drop_duplicates().reset_index(drop=True)
n_states = len(all_states)
print(f"Training with {n_states} distinct states (k={k})")
all_state_index = {tuple(row): idx for idx, row in all_states.iterrows()}
all_state_unabs_index = {tuple(row): idx for idx, row in all_states_unabs.iterrows()}
    
unabs_to_abs_state = {tuple(row[state_cols_simulation]): all_state_index.get(tuple(row[state_cols]), None) 
                      for _, row in df.iterrows()}   
end_time = time.time()
runtime = end_time - start_time 
print(f"Runtime for data preparation: {runtime:.2f} seconds")

cumulative_reward = []
list_len_ep = []
list_granted = []
n_granted = 0
n_stopping_cases = 0
n_stopping_cases_next = 0
optimal_paths = []
start_time = time.time()

#initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

#Q-learning implementation
with tqdm(range(num_episodes), desc="Training Progress", unit="episode") as pbar:
     for episode in pbar:
        reward_episode = 0
        len_epis = 0

        # randomly sampling an event trace
        current_trace_id = np.random.choice(train_case_ids) 
        current_trace = df[df['ID'] == current_trace_id]  
        current_state_unabs = all_state_unabs_index[tuple(current_trace.iloc[0][state_cols_simulation])]
        current_state = all_state_index[tuple(current_trace.iloc[0][state_cols])]
    
        action = 0
       
        Q_table_previous = Q_table
        episode_path = []

        while action not in terminal_actions:
     
            
            possible_actions = [action for action in range(n_actions)
                                if (current_state_unabs, action) in transition_proba and transition_proba[(current_state_unabs, action)].sum() > 0]
            if len(possible_actions) == 0:
                    Q_table = Q_table_previous
                    n_stopping_cases +=1
                    break
           
            #choose action 
            if np.random.rand() < epsilon: # exploration
                action = np.random.choice(possible_actions) # choice restricted to possible actions
                
            else: #exploitation
                max_value_indices = np.where(Q_table[current_state, possible_actions] == np.max(Q_table[current_state, possible_actions]))[0]
                action = possible_actions[np.random.choice(max_value_indices)]
                
            #move to the next unabstraced state
            next_state_unabs, proba = find_next_state(current_state_unabs, action, transition_proba, n_states_unabs)
         
            #get the abstracted version of the next state
            next_state = unabs_to_abs_state.get(tuple(all_states_unabs.iloc[next_state_unabs]), None)
            
            if len_epis >= stopping_criteria or next_state is None: #reverse the episode
                Q_table = Q_table_previous
                n_stopping_cases_next +=1
                break
           
            
            reward =get_reward(dataset, action, budget, cost, activity_index)
            
            episode_path.append(current_state)
            episode_path.append([key for key, value in activity_index.items() if value == action])

            #update Q-value using the Bellman Equation
            Q_table[current_state, action] += learning_rate * (reward + discount_factor *np.max(Q_table[next_state]) - Q_table[current_state, action])
            
            current_state = next_state  
            current_state_unabs = next_state_unabs
            reward_episode += reward
            len_epis += 1


        if dataset == 'bpi2017' and action == activity_index['A_Pending']:
            n_granted +=1
        
        elif dataset == 'bpi2012' and action == activity_index['A_APPROVED']:
            n_granted +=1
        
            
        cumulative_reward.append(reward_episode)
        list_len_ep.append(len_epis)
        list_granted.append(100*(n_granted/(episode+1)))
        episode_path.append(current_state)
        episode_path.append(reward_episode)
        optimal_paths.append(episode_path)

  
        epsilon = max(epsilon_end, epsilon - epsilon_decay) # decay epsilon

        pbar.set_postfix(success=f"{100 * (n_granted / (episode + 1)):.2f}%", len_episode=f"{np.mean(list_len_ep)}")
    
end_time = time.time()
runtime = end_time - start_time 

#transform Q-table into more readable format and save it
all_states_copy = all_states.copy()
all_states_copy['combined'] = all_states_copy[state_cols].apply(lambda row: ','.join(row.astype(str)), axis=1)
Q_table_df = pd.DataFrame(Q_table, index=all_states_copy['combined'], columns=all_actions)



hashable_optimal_paths = [convert_to_tuple(path) for path in optimal_paths[-100:]]
optimal_path_frequencies = Counter(hashable_optimal_paths)


print("=====Training results====")
training_results = {"state_abstraction": state_abstraction,
                    "k": k,
                    "Q_table": Q_table_df,
                    "state_cols": state_cols,
                    "n_states": n_states,
                    "optimal_path_frequencies": optimal_path_frequencies,
                    "cumulative_reward": cumulative_reward,
                    "len_ep": list_len_ep,
                    "granted": list_granted,
                    "n_granted": n_granted,
                    "runtime": runtime,
                    "budget": budget}
print("Training runtime:", runtime)
print(f"Training success rate: {100 * (n_granted / (episode + 1)):.2f}%")


print("=====Testing results====")
testing_results = testing_RL(Q_table, test_case_ids)
print(f"Testing success rate: {100 * (testing_results['n_granted'] / (num_episodes_test + 1)):.2f}%")


os.makedirs("results", exist_ok=True)

training_filename = f"results/{dataset}_{state_abstraction}_{benchmark}_training_{k}.pkl"
with open(training_filename, 'wb') as f:
    pickle.dump(training_results, f)

testing_filename = f"results/{dataset}_{state_abstraction}_{benchmark}_testing_{k}.pkl"
with open(testing_filename, 'wb') as f:
    pickle.dump(testing_results, f)