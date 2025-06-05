import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter
import numpy as np
import os
np.random.seed(0)

#========= Preprocess BPIC12 event log ===========
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'BPI_Challenge_2012.xes.gz')
file_path = os.path.abspath(file_path)
log = xes_importer.apply(file_path)
df = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)

df.rename(columns={'concept:name': 'action', 
                        'case:RequestedAmount': 'amount',
                        'case:concept:name': 'ID',
                        'time:timestamp':'timestamp'}, inplace=True)

#remove incomplete traces
cases_to_remove = []
for case_id, group in df.groupby('ID'):               
        activity_sequence = group['action'].values
        if not any(action in activity_sequence for action in ['A_CANCELLED','A_DECLINED','A_APPROVED'] ):
                        cases_to_remove.append(case_id)
all_cases = df['ID'].unique()
df = df[~df['ID'].isin(cases_to_remove)]

df['time_since_last_act'] = df.groupby('ID')['timestamp'].diff() #get the time between activities
df['time_since_start'] = df.groupby('ID')['timestamp'].transform(lambda x: (x - x.min()).dt.days) #get the time since start of the trace
df['time_since_last_act'] = pd.to_timedelta(df['time_since_last_act'])
df['NumberOfOffers'] = df.groupby('ID')['action'].transform(lambda x: (x == "O_CREATED").cumsum())
case_outcome = df.groupby('ID')['action'].apply(lambda x: 'Success' if 'A_APPROVED' in x.values else 'Failure')
df['Outcome'] = df['ID'].map(case_outcome)

#removing redundant events
mask = (
 
    (df['action'] == df.groupby('ID')['action'].shift(1)) &  #check if the previous action is the same
    (df['time_since_last_act'] < pd.Timedelta(minutes=5))  #check if time difference is less than 5 minutes
)
df = df[~mask].reset_index(drop=True)

hf_mappings = {
    'call#': ['W_Nabellen offertes'],
    'miss#': ['W_Nabellen incomplete dossiers'],
    'offer#': ['O_CREATED'],
    'reply#': ['O_SENT_BACK'],
    'fix': ['W_Wijzigen contractgegevens']
}

for hf_feature in hf_mappings.keys():
    df[hf_feature] = 0
for feature, activities in hf_mappings.items():
    df[feature] = df.groupby('ID')['action'].transform(lambda x: x.isin(activities).cumsum())
df['fix'] = df['fix'].astype(bool)

def classify_amount(amount):
    if amount <= 6000:
        return 'low'
    elif amount > 15000:
        return 'high'
    else:
        return 'medium'

#apply classification to a new column 'amClass' for Branchi et al. (2022) Benchmark
df['case:AMOUNT_REQ'] = pd.to_numeric(df['case:AMOUNT_REQ'], errors='coerce')
df['amClass'] = df['case:AMOUNT_REQ'].apply(classify_amount)
df['last_action'] = df.groupby('ID')['action'].shift(1)
df['last_action'] = df['last_action'].fillna("Start")  
df['event_number'] = df.groupby('ID').cumcount() 

environmental_actions = ['O_CANCELLED', "O_ACCEPTED", 'O_SENT_BACK']
df = df[~df['action'].isin(environmental_actions)]
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))
output_path = os.path.join(data_dir, 'df_bpi2012_preprocessedv2.csv')
df.to_csv(output_path, index=False)


#========= Preprocess BPIC17 event log ===========
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'BPI_Challenge_2017.xes.gz')
file_path = os.path.abspath(file_path)
log = xes_importer.apply(file_path)
df = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)
df.rename(columns={'concept:name': 'action', 
                        'case:LoanGoal': 'goal',
                        'case:ApplicationType': 'type',
                        'case:RequestedAmount': 'amount',
                        'case:concept:name': 'ID',
                        'time:timestamp':'timestamp'}, inplace=True)

#remove incomplete traces
cases_to_remove = []
for case_id, group in df.groupby('ID'):               
        activity_sequence = group['action'].values
        if not any(action in activity_sequence for action in ['A_Cancelled', 'A_Pending', 'A_Denied']):
                        cases_to_remove.append(case_id)

df = df[~df['ID'].isin(cases_to_remove)]
all_cases = df['ID'].unique()

# removing infrequent traces
df['action'] = df['action'].astype(str)  
variants_case = df.groupby('ID')['action'].apply(lambda x: ' -> '.join(x))
trace_variants = variants_case.value_counts().reset_index()
trace_variants.columns = ['trace_variant', 'frequency']
trace_variants['percentage'] = (trace_variants['frequency'] / len(all_cases)) * 100
trace_variants['cumulative_percentage'] = trace_variants['percentage'].cumsum()

top_80_trace_variants = trace_variants[trace_variants['cumulative_percentage'] <= 80]
df['trace_variant'] = df['ID'].map(variants_case)  
df= df[df['trace_variant'].isin(top_80_trace_variants['trace_variant'])]
df = df.drop(columns=['trace_variant'])

df['time_since_last_act'] = df.groupby('ID')['timestamp'].diff() 
df['time_since_start'] = df.groupby('ID')['timestamp'].transform(lambda x: (x - x.min()).dt.days) 
df['time_since_last_act'] = pd.to_timedelta(df['time_since_last_act'])
df['NumberOfOffers'] = df.groupby('ID')['action'].transform(lambda x: (x == "O_Create Offer").cumsum())
case_outcome = df.groupby('ID')['action'].apply(lambda x: 'Success' if 'A_Pending' in x.values else 'Failure')
df['Outcome'] = df['ID'].map(case_outcome)

#identify consecutive duplicate actions that meet the conditions
mask = (
    (df['EventOrigin'] == 'Workflow') & 
    (df['action'] == df.groupby('ID')['action'].shift(1)) &  
    (df['time_since_last_act'] < pd.Timedelta(minutes=5))  #check if time difference is less than 5 minutes
)
df = df[~mask].reset_index(drop=True)

nan_col = ['FirstWithdrawalAmount', 'NumberOfTerms', 'Accepted', 'MonthlyCost',
       'Selected', 'CreditScore', 'OfferedAmount',
       'time_since_start']
for col in nan_col:
    df[col] = df[col].fillna(0)

hf_mappings = {
    'call#': ['W_Call after offers'],
    'miss#': ['W_Call incomplete files'],
    'offer#': ['O_Create Offer'],
    'reply#': ['O_Returned']
}

for hf_feature in hf_mappings.keys():
    df[hf_feature] = 0

for feature, activities in hf_mappings.items():
    df[feature] = df.groupby('ID')['action'].transform(lambda x: x.isin(activities).cumsum())


df['last_action'] = df.groupby('ID')['action'].shift(1)
df['last_action'] = df['last_action'].fillna("Start") 
df['event_number'] = df.groupby('ID').cumcount() 

environmental_actions = ['O_Accepted', 'O_Cancelled', 'O_Returned']
df = df[~df['action'].isin(environmental_actions)]
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))
output_path = os.path.join(data_dir, 'df_bpi2017_preprocessedv2.csv')
df.to_csv(output_path, index=False)