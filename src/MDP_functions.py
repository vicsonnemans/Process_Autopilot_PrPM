import os
import sys
import time
import math
import pickle
from collections import defaultdict, Counter
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cdist
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
np.random.seed(0)

def get_data(dataset, k, state_abstraction):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

        if state_abstraction == 'structural': #transition-based abstraction
            filename = f'df_{dataset}_{k}_structural.csv'
            abstraction_args = ['--state_abstraction=structural', f'--k={k}']
        elif state_abstraction == 'k_means': #full k-means abstraction
            filename = f'df_{dataset}_{k}_clusters.csv'
            abstraction_args = ['--state_abstraction=k_means', f'--k={k}']
        elif state_abstraction == 'k_means_features': #partial k-means abstraction
            filename = f'df_{dataset}_{k}_clusters_features.csv'
            abstraction_args = ['--state_abstraction=k_means_features', f'--k={k}']
        elif state_abstraction == False or state_abstraction == 'last_action': #no abstraction and contextual abstraction
            filename = f'df_{dataset}_preprocessedv2.csv'
            abstraction_args = ['--state_abstraction=last_action']
        else:
            raise ValueError(f"Unsupported state abstraction: {state_abstraction}")

        file_path = os.path.join(data_dir, filename)

        #check if the file exists, otherwise run state_abstraction.py
        if not os.path.exists(file_path):
            print(f"[INFO] File {filename} not found. Running state_abstraction.py...")
            abstraction_args = [f'--state_abstraction={state_abstraction}']
            abstraction_script = os.path.join(script_dir, 'state_abstraction.py')
            if k is not None:
                abstraction_args.append(f'--k={k}')

            subprocess.run(['python', abstraction_script] + abstraction_args + [f'--dataset={dataset}'], check=True)

        df = pd.read_csv(file_path, sep=",")

        all_cases = df['ID'].unique()
        all_actions = df["action"].unique() 
        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        n_actions = len(all_actions)
        df_success = df[df['Outcome']=='Success']

        # compute the budget
        trace_lengths = df_success.groupby("ID")["action"].count()
        third_quantile = np.percentile(trace_lengths, 75)
        budget = math.ceil(third_quantile)

        df["remaining_budget"] = budget - df["event_number"]
        df['available_budget'] = (df['remaining_budget'] > 0).astype(int)
        
        return df, df_success, all_cases, all_actions, activity_index, n_actions, budget



def find_next_state(current_state, action, transition_proba, n_states_unabs): #using unabstracted original states


        prob_vector = transition_proba.get((current_state, action), None).toarray().flatten()
        states = np.arange(n_states_unabs)
        next_state = np.random.choice(states, size=1, p=prob_vector)[0] 
        proba = prob_vector[next_state]
                
        return next_state, proba

def transition_probabilities_faster(df, state_cols, all_states, activity_index, n_actions):

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

            
                for i in range(len(state_sequence) - 1):
          
                    current_state = state_index_map[
                            (tuple(state_sequence[i][col] for col in state_cols))
                    ]
                    next_state = state_index_map[
                            (tuple(state_sequence[i + 1][col] for col in state_cols))
                    ]
                    current_action = activity_index[action_sequence[i]]

                    state_action_freq[current_state, current_action] += 1
                    if (current_state, current_action) not in state_freq:
                        state_freq[(current_state, current_action)] = dok_matrix((n_states, 1), dtype=np.float64)
                    state_freq[(current_state, current_action)][next_state, 0] += 1

    
    transition_proba = {}  

    for (state, action), freq_matrix in state_freq.items():
        if state_action_freq[state, action] > 0:
            transition_proba[(state, action)] = freq_matrix / state_action_freq[state, action]

    return transition_proba



def get_reward(dataset, current_action,budget, cost, activity_index):

    if dataset == 'bpi2017':
        if current_action == activity_index['A_Pending']: #positive terminal state
            reward = budget            
        else: 
            reward = -cost #intermediate action

    elif dataset == 'bpi2012':
        if current_action == activity_index['A_APPROVED']: #positive terminal state
            reward = budget         
        else: 
            reward = -cost #intermediate action

        
    return reward

def define_state_cols(dataset, df, k, state_abstraction, benchmark, activity_index):
    if dataset == 'bpi2017':
        features = ['goal', 'type', 'amount','FirstWithdrawalAmount', 'NumberOfTerms', 'Accepted', 'MonthlyCost','Selected', 'CreditScore', 'OfferedAmount', 'time_since_start']
        control_var = ['call#', 'miss#', 'offer#', 'reply#']
        terminal_activities = [activity_index['A_Pending'], activity_index['A_Cancelled'], activity_index['A_Denied']]

    elif dataset == 'bpi2012':
        features = ['case:AMOUNT_REQ', 'time_since_start']
        control_var = ['call#', 'miss#', 'offer#', 'reply#', 'fix']
        terminal_activities = [activity_index['A_APPROVED'], activity_index['A_CANCELLED'], activity_index['A_DECLINED']] 

    state_cols_simulation = ['last_action'] + features + control_var
    
    if state_abstraction == 'k_means_features': #partial k-means abstraction
        state_cols = ['last_action', 'cluster']+ control_var
    
    elif state_abstraction == 'k_means': #full k-means abstraction
        state_cols = ['last_action', 'cluster']
    
    elif state_abstraction == 'structural': #transition-based abstraction
        state_cols = ['cluster']

    elif state_abstraction == False:  #no state abstraction
        state_cols = state_cols_simulation
    
    elif state_abstraction == 'last_action': #contextual abstraction
        state_cols = ['last_action']

    return state_cols, state_cols_simulation, terminal_activities



