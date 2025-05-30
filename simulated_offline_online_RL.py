from MDP_functions import get_data, find_next_state, transition_probabilities_faster
from utils import *
from MDP_generator import *
import os
np.random.seed(0)

dataset = 'simulation'
num_episodes_online = 30000
num_times = 5
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
                state_key = tuple(all_states_unabs.iloc[current_state_unabs][state_cols_simulation])
                reward = get_reward_simulation(action)
                
                episode_path.append(current_state)
                episode_path.append([key for key, value in activity_index.items() if value == action])

                #update Q-value using the Q-learning update rule
                Q_table[current_state, action] += learning_rate * (reward + discount_factor *np.max(Q_table[next_state]) - Q_table[current_state, action])
                
                current_state = next_state  
                current_state_unabs = next_state_unabs
                reward_episode += reward
                len_epis += 1        
            
            if dataset == 'simulation' and action == activity_index['approve']:
                n_granted_test +=1

            cumulative_reward_test.append(reward_episode)
            list_granted_test.append(100*(n_granted_test/(episode+1)))
            list_len_ep.append(len_epis)
            episode_path.append(current_state)
            episode_path.append(reward_episode)
            optimal_paths.append(episode_path)

            

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

print("=========loading environment==========")
env = InterpretableLoanMDP()
dataset = 'simulation'
benchmark = False

print("=========Peforming state abstraction==========")
state_cols, state_cols_simulation, terminal_actions = define_state_cols_sim(False, k)
df, df_success, all_cases, all_actions, activity_index_df, n_actions, budget = get_data(dataset, k, False)
activity_index = {v: k for k, v in env.activity_meanings.items()}

if state_abstraction == 'k_means':
    clustering_state_cols = ['loan_amount','credit_score','debt_to_income','purpose', 'num_calls','num_offers','offer_amount','customer_response_to_call','customer_acceptance_of_offer','customer_cancellation']
    df, kmeanModel = k_means(df, clustering_state_cols, k)
    df.to_csv(f'df_simulation_{k}_clusters.csv', index=False)

elif state_abstraction == 'k_means_features':
    clustering_state_cols = ['loan_amount','credit_score','debt_to_income','purpose']
    df, kmeanModel = k_means(df, clustering_state_cols, k)
    df.to_csv(f'df_simulation_{k}_clusters_features.csv', index=False)


elif state_abstraction == 'structural':

    #unabstracted MDP - get the transitions
    all_states_unabs = df[state_cols_simulation].drop_duplicates().reset_index(drop=True)
    all_states = df[state_cols].drop_duplicates().reset_index(drop=True)
    transition_proba = transition_probabilities_faster_2(df, state_cols_simulation, all_states_unabs, activity_index, n_actions)
    all_states_unabs = all_states_unabs.drop(columns=["state_index"])

    df_structural = df.copy()
    all_states['state_index'] = all_states.index
    state_index_map = {
                    (tuple(row[state_cols])): row['state_index']
                    for _, row in all_states.iterrows()
                }
    df_structural["state"] = df_structural[state_cols].apply(lambda row: state_index_map.get(tuple(row), -1), axis=1)
    df_structural["next_state"] = df_structural.groupby("ID")["state"].shift(-1).fillna(-1).astype(int)
    
    #build transition sets per state
    state_action_map = defaultdict(set)
    for _, row in df_structural.iterrows():
        state_action_map[row["state"]].add(f"{row['action']};{row['next_state']}")

    #identify unique state groups
    unique_state_groups = {}
    merged_state_index = 0
    state_to_cluster = {}

    for state, action_next_pairs in state_action_map.items():
            action_next_pairs = tuple(sorted(action_next_pairs))
            if action_next_pairs not in unique_state_groups:
                unique_state_groups[action_next_pairs] = merged_state_index
                merged_state_index += 1
            state_to_cluster[state] = unique_state_groups[action_next_pairs]  

    df_structural["cluster"] = df_structural["state"].map(state_to_cluster)
    df_structural.to_csv(f'df_{dataset}_{k}_structural.csv', index=False)
    
print("=========Offline Reinforcement Learning==========") 

#split into train and test set
shuffled_cases = np.random.permutation(np.arange(0, len(all_cases), 1))
train_size = int(0.8 * len(all_cases))
train_case_ids = shuffled_cases[:train_size] 
test_case_ids = shuffled_cases[train_size:] 

#define the hyperparameters 
num_episodes = int(len(train_case_ids) * num_times)
num_episodes_test = len(test_case_ids)
learning_rate = 0.8
discount_factor = 0.9
stopping_criteria = 100 #max length of traces
cost = 1 #cost of individual action
epsilon_start = 0.1
epsilon_end = 0.01
decay_rate = np.log(epsilon_end / epsilon_start) / num_episodes
epsilon = epsilon_start


print(f"=====Data preparation for {dataset}, state abstraction: {state_abstraction}, k: {k}====")
state_cols, state_cols_simulation, terminal_actions = define_state_cols_sim(state_abstraction, k)
print(f"Abstracted state columns: {state_cols}") #abstracted state columns
epsilon = epsilon_start
start_time = time.time()
df, df_success, all_cases, all_actions, activity_index_df, n_actions, budget = get_data(dataset, k, state_abstraction)
activity_index = {v: k for k, v in env.activity_meanings.items()}
all_states_unabs = df[state_cols_simulation].drop_duplicates().reset_index(drop=True)
n_states_unabs = len(all_states_unabs)
print(f"Simulating the environment with {n_states_unabs} distinct original states")
transition_proba = transition_probabilities_faster_2(df, state_cols_simulation, all_states_unabs, activity_index, n_actions)
all_states_unabs = all_states_unabs.drop(columns=["state_index"])
all_states = df[state_cols].drop_duplicates().reset_index(drop=True)
n_states = len(all_states)
print(f"Training with {n_states} distinct abstracted states (k={k})")
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
list_success = []
start_time = time.time()

#initialize Q-table with zeros
Q_table_offline = np.zeros((n_states, len(activity_index)))

#Q-learning implementation
with tqdm(range(num_episodes), desc="Training Progress", unit="episode") as pbar:
     for episode in pbar:
        reward_episode = 0
        len_epis = 0
        success=0

        # randomly sampling an event trace
        current_trace_id = np.random.choice(train_case_ids) 
        current_trace = df[df['ID'] == current_trace_id]  
        current_state_unabs = all_state_unabs_index[tuple(current_trace.iloc[0][state_cols_simulation])]
        current_state = all_state_index[tuple(current_trace.iloc[0][state_cols])]
    
        action = 0
       
        Q_table_previous = Q_table_offline
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
                
            else: # exploitation
                max_value_indices = np.where(Q_table_offline[current_state, possible_actions] == np.max(Q_table_offline[current_state, possible_actions]))[0]
                action = possible_actions[np.random.choice(max_value_indices)]
                
            #move to the next unabstraced state
            next_state_unabs, proba = find_next_state(current_state_unabs, action, transition_proba, n_states_unabs)
         
            #get the abstracted version of the next state
            next_state = unabs_to_abs_state.get(tuple(all_states_unabs.iloc[next_state_unabs]), None)
            
            if len_epis >= stopping_criteria or next_state is None: #reverse the episode
                Q_table_offline = Q_table_previous
                n_stopping_cases_next +=1
                break
           

            state_key = tuple(all_states_unabs.iloc[current_state_unabs][state_cols_simulation])
            reward = get_reward_simulation(action)

            
            episode_path.append(tuple(all_states.iloc[current_state][state_cols]))
            episode_path.append([key for key, value in activity_index.items() if value == action])

            #update Q-value using the Bellman Equation
            Q_table_offline[current_state, action] += learning_rate * (reward + discount_factor *np.max(Q_table_offline[next_state]) - Q_table_offline[current_state, action])
            
            current_state = next_state  
            current_state_unabs = next_state_unabs
            reward_episode += reward
            len_epis += 1


        if dataset == 'simulation' and action == activity_index['approve']:
            n_granted +=1
            success = 1
                  
        cumulative_reward.append(reward_episode)
        list_len_ep.append(len_epis)
        list_success.append(success)
        list_granted.append(100*(n_granted/(episode+1)))
        episode_path.append(current_state)
        episode_path.append(reward_episode)
        optimal_paths.append(episode_path)

        # Exponential decay of epsilon
        epsilon = epsilon_start * np.exp(decay_rate * episode)
        epsilon = max(epsilon_end, epsilon)

        pbar.set_postfix(success=f"{100 * (n_granted / (episode + 1)):.2f}%", len_episode=f"{int(np.mean(list_len_ep))}")
    
end_time = time.time()
runtime = end_time - start_time 

hashable_optimal_paths = [convert_to_tuple(path) for path in optimal_paths[-100:]]
optimal_path_frequencies = Counter(hashable_optimal_paths)



print("=====Training results====")
training_results = {"state_abstraction": state_abstraction,
                    "Q_table": Q_table_offline,
                    "k": k,
                    "state_cols": state_cols,
                    "n_states": n_states,
                    "optimal_path_frequencies": optimal_path_frequencies,
                    "cumulative_reward": cumulative_reward,
                    "list_success": list_success,
                    "len_ep": list_len_ep,
                    "granted": list_granted,
                    "n_granted": n_granted,
                    "runtime": runtime,
                    "budget": budget}
print("Training runtime:", runtime)
print(f"Training success rate: {100 * (n_granted / (episode + 1)):.2f}%")


print("=====Testing results====")
testing_results = testing_RL(Q_table_offline, test_case_ids)
print(f"Testing success rate: {100 * (testing_results['n_granted'] / (num_episodes_test + 1)):.2f}%")

os.makedirs("results", exist_ok=True)

training_filename = f"results/{dataset}_{state_abstraction}_{benchmark}_training_{k}.pkl"
with open(training_filename, 'wb') as f:
    pickle.dump(training_results, f)

testing_filename = f"results/{dataset}_{state_abstraction}_{benchmark}_testing_{k}.pkl"
with open(testing_filename, 'wb') as f:
    pickle.dump(testing_results, f)

print("=========Online Reinforcement Learning==========") 
def seen_offline(state):
    processed_state = list(state)
    last_action_idx = int(processed_state[-1])
    last_action_str = env.activity_meanings.get(last_action_idx, f"unknown_{last_action_idx}")
    processed_state[-1] = last_action_str  # Replace numeric index with action name
 
    if tuple(flatten_and_convert(processed_state)) not in all_state_unabs_index.keys(): #novel case (not in event log)
        return True
    else: 
        return False
    
def get_structural_state(processed_state, state_index_map_extended, state_to_cluster_extended):
    #get current state index
    flat_processed_state = flatten_and_convert(processed_state)
    state_tuple = tuple(flat_processed_state)
    if state_tuple not in state_index_map_extended: #unvisted state
        state_idx = len(state_index_map_extended)  
        state_index_map_extended[state_tuple] = state_idx

    else: #visited state
        state_idx = state_index_map_extended[state_tuple]
  
    if state_idx not in state_to_cluster_extended: #create a new one if state is unvisited
        cluster_id = max(state_to_cluster_extended.values()) + 1
        state_to_cluster_extended[state_idx] = cluster_id
    else:#retrieve the cluster ID for the current state
        cluster_id = state_to_cluster_extended[state_idx]
    
    processed_state = [cluster_id]
    return processed_state, state_index_map_extended, state_to_cluster_extended

def get_state_index(env, state, Q_table, all_states_extended, all_state_index_extended, state_index_map_extended, state_to_cluster_extended):
   
    processed_state = list(state)
    last_action_idx = int(processed_state[-1])
    last_action_str = env.activity_meanings.get(last_action_idx, f"unknown_{last_action_idx}")
    processed_state[-1] = last_action_str  #eeplace numeric index with action name
    
    if state_abstraction == 'k_means' or state_abstraction=='k_means_features':
        processed_state = get_kmeans_state(processed_state, clustering_state_cols, kmeanModel)
    elif state_abstraction == 'structural':
        processed_state, state_index_map, state_to_cluster = get_structural_state(processed_state, state_index_map_extended, state_to_cluster_extended)
    elif state_abstraction == 'last_action':
        state_str = last_action_str
        state_tuple = (state_str,)

    if state_abstraction != 'last_action':
        flat_processed_state = flatten_and_convert(processed_state)
        state_tuple = tuple(flat_processed_state)
    
    if state_tuple in all_state_index_extended:
        unseen_state = False
        return all_state_index_extended[state_tuple], unseen_state, Q_table, all_states_extended, all_state_index_extended

    new_row = pd.DataFrame([list(state_tuple)], columns=state_cols)
    all_states_extended = pd.concat([all_states_extended, new_row], ignore_index=True)
    new_index = len(all_states_extended) - 1

    zero_row = np.zeros(Q_table.shape[1])
    Q_table = np.vstack([Q_table, zero_row])
    all_state_index_extended[state_tuple] = new_index
    unseen_state = True

    return new_index, unseen_state, Q_table, all_states_extended, all_state_index_extended

#define the hyperparameters 
num_episodes = num_episodes_online
epsilon_start = 0.1
epsilon_end = 0.01
decay_rate = np.log(epsilon_end / epsilon_start) / num_episodes
epsilon = epsilon_start

Q_table = Q_table_offline.copy()
all_states_extended = all_states.copy()
all_state_index_extended = all_state_index.copy()
if state_abstraction == 'structural':
    state_index_map_extended = state_index_map.copy()
    state_to_cluster_extended = state_to_cluster.copy()

else: 
    state_index_map_extended = None
    state_to_cluster_extended = None
n_states = Q_table.shape[0] #offline number of states

print(f"=====Online learning for {dataset}, state abstraction: {state_abstraction}, k: {k}====")

cumulative_reward = []
list_len_ep = []
list_granted = []
list_success = []
n_granted = 0
n_stopping_cases = 0
n_stopping_cases_next = 0
optimal_paths = []
incorrect_transitions = 0
n_unseen_cases = 0
n_unseen_granted_cases = 0
n_new_states = 0
n_unmapped_states = 0
n_new_starts = 0
n_seen_cases = 0 
n_seen_granted_cases = 0
total_visited_states = 0
start_time = time.time()


#Q-learning implementation
with tqdm(range(num_episodes), desc="Online Training Progress", unit="episode") as pbar:
     for episode in pbar:
        reward_episode = 0
        len_epis = 0
        success = 0
        done = False
        granted = False
        unseen_start = False
        new_trajectory = False

        episode_path = []

        # randomly sampling a starting state
        state = env.reset()
    
        
        state_id, unmapped_state, Q_table, all_states_extended, all_state_index_extended = get_state_index(env, state, Q_table, all_states_extended, all_state_index_extended, state_index_map_extended, state_to_cluster_extended)
        #unmapped state is true if the state has created new row in Q-table

        processed_state = list(state)
        last_action_idx = int(processed_state[-1])
        last_action_str = env.activity_meanings.get(last_action_idx, f"unknown_{last_action_idx}")
        processed_state[-1] = last_action_str  #replace numeric index with action name
 
        if tuple(flatten_and_convert(processed_state)) not in all_state_unabs_index.keys(): #novel case (not in event log)
            n_new_starts += 1
            unseen_start = True
            new_trajectory = True

        while not done and len_epis < stopping_criteria:
           
            possible_actions = env.get_valid_actions()
            new_state = seen_offline(state)

            if new_state:
                n_new_states += 1 
                new_trajectory = True
            if unmapped_state:
                n_unmapped_states += 1

            # choose action 
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n) # exploration
            else:
                max_value_indices = np.where(Q_table[state_id] == np.max(Q_table[state_id]))[0]
                action = np.random.choice(max_value_indices)
            
                if action not in possible_actions: 
                    incorrect_transitions += 1

            #take action in env
            next_state, reward, done, granted, info = env.step(action)
            next_state_id, unmapped_state, Q_table, all_states_extended, all_state_index_extended = get_state_index(env,next_state, Q_table, all_states_extended, all_state_index_extended, state_index_map_extended, state_to_cluster_extended) #for online Q-table 

            episode_path.append(state)
            episode_path.append([key for key, value in activity_index.items() if value == action])

            #update Q-value using the Bellman Equation
            Q_table[state_id, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state_id]) - Q_table[state_id, action])
            
            state = next_state  
            state_id = next_state_id
            previous_action = action
       
            reward_episode += reward
            len_epis += 1
            
            total_visited_states += 1
           

        if not new_trajectory:
            n_seen_cases += 1
            if granted:
                n_seen_granted_cases += 1
                n_granted +=1
                success = 1
        if new_trajectory:
            n_unseen_cases += 1
            if granted:
                n_unseen_granted_cases += 1 #to assess generalization
                n_granted +=1
                success = 1

            
        cumulative_reward.append(reward_episode)
        list_len_ep.append(len_epis)
        list_granted.append(100*(n_granted/(episode+1)))
        episode_path.append(state)
        episode_path.append(reward_episode)
        optimal_paths.append(episode_path)
        list_success.append(success)
     
        epsilon = epsilon_start * np.exp(decay_rate * episode)
        epsilon = max(epsilon_end, epsilon)

        pbar.set_postfix(success=f"{100 * (n_granted / (episode + 1)):.2f}%", len_episode=f"{int(np.mean(list_len_ep))}")
    
end_time = time.time()
runtime = end_time - start_time 

print(np.mean(list_len_ep[-100:]))


print("=====Online Learning results====")
training_results = {"state_abstraction": state_abstraction,
                    "k": k,
                    "state_cols": state_cols,
                    "n_states_offline": n_states,
                    "n_states_online": Q_table.shape[0],
                    'n_new_states': n_new_states,
                    "n_unmapped_states": n_unmapped_states,
                    "total_visited_states": total_visited_states,
                    'incorrect_transitions': incorrect_transitions,
                    "n_unseen_granted_cases": n_unseen_granted_cases,
                    "n_seen_granted_cases": n_seen_granted_cases,
                    "n_unseen_cases": n_unseen_cases,
                    "n_seen_cases": n_seen_cases,
                    "optimal_path_frequencies": optimal_path_frequencies,
                    "cumulative_reward": cumulative_reward,
                    "len_ep": list_len_ep,
                    "granted": list_granted,
                    "n_granted": n_granted,
                    "list_success": list_success,
                    "runtime": runtime,
                    "budget": budget}
print("Online Training runtime:", runtime)
print(f"Online success rate: {100 * (n_granted / (episode + 1)):.2f}%")


os.makedirs("results", exist_ok=True)

training_filename = f"results/{dataset}_{state_abstraction}_{benchmark}_online_{k}.pkl"
with open(training_filename, 'wb') as f:
    pickle.dump(training_results, f)