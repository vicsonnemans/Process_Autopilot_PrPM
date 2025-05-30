# === Standard library ===
import sys
import os
import time
import math
import pickle
import random
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations

# === Data manipulation and visualization ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import seaborn as sns
from scipy.sparse import dok_matrix
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

# === Scikit-learn ===
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# === Gym and environments ===
import gymnasium as gym
from gym import spaces

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# === Custom utility ===
from scipy.stats import wasserstein_distance_nd

control_var = ['num_calls','num_offers','offer_amount','customer_response_to_call','customer_acceptance_of_offer','customer_cancellation']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def define_state_cols_sim(state_abstraction, k):
        features = ['loan_amount','credit_score','debt_to_income','purpose']
        control_var = ['num_calls','num_offers','offer_amount','customer_response_to_call','customer_acceptance_of_offer','customer_cancellation']
        state_cols_simulation = features + control_var + ['last_action']
        termination_actions = [6,7,8] #approve, reject, cancellation received
        
        if state_abstraction == 'k_means':
                state_cols =  ['cluster', 'last_action']
        
        elif state_abstraction == 'k_means_features':
                state_cols =  ['cluster'] + control_var + ['last_action']
        
        elif state_abstraction == 'structural': #stochastic bisimulation
                state_cols = ['cluster']
        
        elif state_abstraction == 'last_action':
                state_cols = ['last_action']
                
        elif state_abstraction == False:  #no state abstraction
                state_cols = state_cols_simulation

        return state_cols, state_cols_simulation, termination_actions

def flatten_and_convert(state):
        flat = []
        for v in state:
            if isinstance(v, (list, tuple, np.ndarray)):
                flat.extend(flatten_and_convert(v))  # recursive flatten
            elif isinstance(v, (np.str_, str)):
                flat.append(str(v))
            elif isinstance(v, (np.integer, int)):
                flat.append(int(v))
            elif isinstance(v, (np.floating, float)):
                flat.append(float(v))
            else:
                flat.append(v)
        return flat

def convert_to_tuple(path):
    return tuple(tuple(x) if isinstance(x, list) else x for x in path)


def transform_data(df, clustering_state_cols):
    state_data = df[clustering_state_cols]
    x = pd.get_dummies(state_data)
    x = x.values.reshape(-1, x.shape[1]).astype('float32')
    standardizer = preprocessing.StandardScaler()
    x_train = standardizer.fit_transform(x)
    x_train = torch.from_numpy(x_train).to(device)
    return x_train, standardizer



#====== FOR OFFLINE PHASE ===========

def get_reward_simulation(action):
    reward_table = {
        (6): +10.0, #approve
        (7): -10.0, #reject
        (8): -20.0, #cancellation_received
        (0): 0.0, #receive application
        (1): -1.0, #request docs
        (2): -1.0, #escalate
        (3): -1.0, #create offer
        (4): -1.0, #schedule call
        (5): -1.0 #flag manual
    }
    return reward_table.get((action), -3.0)

def k_means(df, state_cols, k):

    df_selected = df[state_cols]
    categorical_columns = df_selected.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df_selected[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    df_selected = df_selected.reset_index(drop=True)
    one_hot_df = one_hot_df.reset_index(drop=True)
    df_encoded = pd.concat([df_selected, one_hot_df], axis=1)
    state_cols = list(df_encoded.drop(categorical_columns, axis=1).columns) 
    df_encoded = df_encoded.drop(categorical_columns, axis=1)
    df_encoded[state_cols].drop_duplicates().reset_index(drop=True)

    #apply k-means clustering on the event attributes
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(df_encoded)
    df_encoded['cluster'] = kmeanModel.labels_
    df.reset_index(drop=True, inplace=True)
    df['cluster']=df_encoded['cluster'] #add clusters to df
    #all_states_abs = df[['last_action',"cluster"]+control_var].drop_duplicates().reset_index(drop=True)
    all_states_abs = df["cluster"].drop_duplicates().reset_index(drop=True)
    n_states_abs = len(all_states_abs)

    print(f"Number of distinct states for {k} clusters: {n_states_abs}")

    return df, kmeanModel


#====== FOR ONLINE PHASE ===========
def get_kmeans_state(processed_state, clustering_state_cols, kmeanModel):
    df_state = pd.DataFrame([processed_state[:len(clustering_state_cols)]], columns=clustering_state_cols)
    cluster = int(kmeanModel.predict(df_state[clustering_state_cols])[0])
    abs_state = [cluster, processed_state[len(clustering_state_cols):]]
    return abs_state


def generate_event_log(env, num_episodes=1000, max_steps=20, seed=42):
    rng = np.random.default_rng(seed)
    logs = []
    base_time = datetime.now()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        timestamp = base_time + timedelta(seconds=ep * 60)
        step = 0

        # Log initial state before any action
        logs.append({
            "ID": ep,
            "timestamp": timestamp,
            "action":'Start',
            "reward": 0,
            "creditworthy": env.creditworthy,
            **dict(zip(env.feature_names, state.tolist()))
        })

        logs.append({
            "ID": ep,
            "timestamp": timestamp,
            "action": env.activity_meanings[env.last_action],
            "reward": 0,
            "creditworthy": env.creditworthy,
            **dict(zip(env.feature_names, state.tolist()))
        })

        step += 1

        while not done and step <= max_steps:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break  # No valid action available

            action = rng.choice(valid_actions)
            next_state, reward, done, granted, info = env.step(action)

            logs.append({
                "ID": ep,
                "timestamp": timestamp + timedelta(seconds=step * 5),
                "action": info["action_taken"],
                "reward": reward,
                "creditworthy": info["creditworthy"],
                **info["features"]
            })

            state = next_state
            step += 1
        
        logs.append({
                "ID": ep,
                "timestamp": timestamp,
                "action": "end",
                "reward": reward,
                "creditworthy": info["creditworthy"],
                **info["features"]
            })

    return pd.DataFrame(logs)

def transition_probabilities_faster_2(df, state_cols, all_states, activity_index, n_actions):

    all_states['state_index'] = all_states.index
    state_index_map = {
                (tuple(row[state_cols])): int(row['state_index'])
                for _, row in all_states.iterrows()
            }


    n_states = len(all_states)
        

    state_action_freq = dok_matrix((n_states, n_actions), dtype=np.float64)
    state_freq = {}

    for case_id, group in tqdm(df.groupby('ID'), desc="Processing Cases"):
                group = group.reset_index(drop=True)

                state_sequence = group[state_cols].to_records(index=False)
                action_sequence = group['action'].values

            
                for i in range(len(state_sequence) - 2):
          
                    current_state = state_index_map[
                            (tuple(state_sequence[i][col] for col in state_cols))
                    ]
                    next_state = state_index_map[
                            (tuple(state_sequence[i + 1][col] for col in state_cols))
                    ]
                    current_action = activity_index[action_sequence[i+1]]
                

                    state_action_freq[current_state, current_action] += 1
                    if (current_state, current_action) not in state_freq:
                        state_freq[(current_state, current_action)] = dok_matrix((n_states, 1), dtype=np.float64)
                    state_freq[(current_state, current_action)][next_state, 0] += 1

    
    transition_proba = {}  

    for (state, action), freq_matrix in state_freq.items():
        if state_action_freq[state, action] > 0:
            transition_proba[(state, action)] = freq_matrix / state_action_freq[state, action]

    return transition_proba

#
def get_transitions_and_rewards(current_state, action, transition_proba):
    

    transition = transition_proba.get((current_state, action), None)
    if transition is None:
        return None, None, None  
    
    prob_vector = transition.toarray().flatten()
    next_states = np.arange(len(prob_vector))
    nonzero_mask_i = prob_vector > 0
    prob_vector = prob_vector[nonzero_mask_i]
    next_states = next_states[nonzero_mask_i]
    reward = get_reward_simulation(action)
    
    return next_states, prob_vector , reward
