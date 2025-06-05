from MDP_functions import *
import argparse
np.random.seed(0)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '..', 'data'))

#parse parameters:
parser = argparse.ArgumentParser(description="Run state abstraction for a given dataset.")
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., bpi2012, bpi2017)')
parser.add_argument('--state_abstraction', type=str, required=True, choices=['k_means', 'k_means_features', 'structural'], help='State abstraction method')
parser.add_argument('--k', type=int, default=None, help='Number of clusters (only required for k-means)')

args = parser.parse_args()

dataset = args.dataset
state_abstraction = args.state_abstraction
k = args.k

#input file and columns
input_file = os.path.join(data_dir, f'df_{dataset}_preprocessedv2.csv')
df = pd.read_csv(input_file, sep=",")

if dataset == 'bpi2017':
    control_var = ['call#', 'miss#', 'offer#', 'reply#']
    if state_abstraction == 'k_means_features':
        state_cols = ['goal', 'type', 'amount', 'NumberOfTerms',
        'FirstWithdrawalAmount', 'Accepted', 'MonthlyCost',
        'Selected', 'CreditScore', 'OfferedAmount', 'time_since_start']
    elif state_abstraction == 'k_means':
        state_cols  = ['goal', 'type', 'amount', 'NumberOfTerms',
        'FirstWithdrawalAmount', 'Accepted', 'MonthlyCost',
        'Selected', 'CreditScore', 'OfferedAmount', 'time_since_start'] + ['call#', 'miss#', 'offer#', 'reply#']
    
elif dataset == 'bpi2012':
    control_var = ['call#', 'miss#', 'offer#', 'reply#', 'fix']
    if state_abstraction == 'k_means_features':
        state_cols = ['case:AMOUNT_REQ', 'time_since_start']
    elif state_abstraction == 'k_means':
        state_cols = ['case:AMOUNT_REQ', 'time_since_start']+['call#', 'miss#', 'offer#', 'reply#', 'fix']
    

def k_means(df, state_cols, k):

    all_states_abs = df[["last_action"] + state_cols + control_var].drop_duplicates().reset_index(drop=True)
    n_states_abs = len(all_states_abs)

    print(f"Number of distinct states for {state_cols}: {n_states_abs}")
    df_selected = df[state_cols]

    #convert categorical variables
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
    all_states_abs = df[['last_action',"cluster"]+control_var].drop_duplicates().reset_index(drop=True)
    n_states_abs = len(all_states_abs)

    print(f"Number of distinct states for {k} clusters: {n_states_abs}")

    return df

#K-means abstraction
if state_abstraction == 'k_means': #full k-means abstraction
    df_abs = k_means(df, state_cols, k)
    output_file = os.path.join(data_dir, f'df_{dataset}_{k}_clusters.csv')
    df_abs.to_csv(output_file, index=False)
elif state_abstraction == 'k_means_features': #partial k-means abstraction
    df_abs = k_means(df, state_cols, k)
    output_file = os.path.join(data_dir, f'df_{dataset}_{k}_clusters_features.csv')
    df_abs.to_csv(output_file, index=False)

#Structural abstraction
if state_abstraction == 'structural':
    #retrieve the unabstracted MDP
    state_abstraction = False
    k = None
    benchmark = False
    df, df_success, all_cases, all_actions, activity_index, n_actions, budget = get_data(dataset, k, state_abstraction)
    state_cols, state_cols_simulation, terminal_actions = define_state_cols(dataset, df, k, state_abstraction, benchmark, activity_index)
    all_states = df[state_cols].drop_duplicates().reset_index(drop=True)
    n_states = len(all_states)
    print(f"{n_states} distinct states in original state space")
    transition_proba = transition_probabilities_faster(df, state_cols, all_states, activity_index, n_actions)

    all_states['state_index'] = all_states.index
    state_index_map = {
                    (tuple(row[state_cols])): row['state_index']
                    for _, row in all_states.iterrows()
                }
    df["state"] = df[state_cols].apply(lambda row: state_index_map.get(tuple(row), -1), axis=1)
    df["next_state"] = df.groupby("ID")["state"].shift(-1).fillna(-1).astype(int)

    state_action_map = defaultdict(set)

    for _, row in df.iterrows():
        state_action_map[row["state"]].add((row["action"], row["next_state"]))

    #identify unique state groups
    unique_state_groups = {}
    merged_state_index = 0

    for state, action_next_pairs in state_action_map.items():
        action_next_pairs = tuple(sorted(action_next_pairs))
        
        #assign a merged state index if not already assigned
        if action_next_pairs not in unique_state_groups:
            unique_state_groups[action_next_pairs] = merged_state_index
            merged_state_index += 1

    #create a new column for merged states
    df["cluster"] = df["state"].map(lambda s: unique_state_groups.get(tuple(sorted(state_action_map.get(s, set()))), -1))

    print(f"Abstracted state size: {len(df['cluster'].unique())}")
    k = None
    output_file = os.path.join(data_dir, f'df_{dataset}_{k}_structural.csv')
    df.to_csv(output_file, index=False)