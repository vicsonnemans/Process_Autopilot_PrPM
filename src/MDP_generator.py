from utils import *
np.random.seed(0)


class InterpretableLoanMDP(gym.Env):
    def __init__(self, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        #static features
        self.feature_config = {
            "loan_amount": (1000, 40000), #loan amount requested
            "credit_score": (300, 850), #applicant's credit score
            "debt_to_income": (0, 100), #applicant's debt-to-income ratio
            "purpose": (0, 3), #loan goal
        }

       
        self.static_features = list(self.feature_config.keys())
    
        #dynamic features
        self.dynamic_features = [
            "num_calls", #number of calls made by the bank since start of case
            "num_offers", #number of offers made by the bank since start of case
            "offer_amount", #amount offered by the bank (=0 if no offers have been created yet)
            "customer_response_to_call",   # 0 or 1
            "customer_acceptance_of_offer",# 0=no/reject,1=accept
            "customer_cancellation"        # 0=no,1=cancelled
        ]
        self.feature_names = self.static_features + self.dynamic_features + ['last_action']
        self.static_dim = len(self.static_features)
        self.dynamic_dim = len(self.dynamic_features)
        self.state_dim = self.static_dim + self.dynamic_dim

        #actions
        self.activity_meanings = {
            0: "receive_application",  
            1: "request_docs",
            2: "escalate",
            3: "create_offer",
            4: "schedule_call",
            5: "flag_manual",
            6: "approve",
            7: "reject",
            8: "cancellation_received"  
        }
    
        self.action_space = spaces.Discrete(len(self.activity_meanings))

        #observation space updated with new dynamic feature ranges
        static_low = np.array([v[0] for v in self.feature_config.values()], dtype=np.int32)
        static_high = np.array([v[1] for v in self.feature_config.values()], dtype=np.int32)
        dynamic_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        dynamic_high = np.array([1000, 1000, 100000, 1, 1, 1], dtype=np.int32)
        last_action_low = np.array([0])
        last_action_high = np.array([len(self.activity_meanings) - 1])

        self.observation_space = spaces.Box(
            low=np.concatenate([static_low, dynamic_low, last_action_low]),
            high=np.concatenate([static_high, dynamic_high, last_action_high]),
            dtype=np.int32
        )

     
        self.last_action = 0

        #allowable transitions by action index (additional conditions and stochasticity are defined in .step() function)
        self.allowed_transitions = {
            0: [1, 5, 8],       #receive_application → request_docs or flag_manual
            1: [2, 3, 4, 8],    #request_docs → escalate, create_offer, or schedule_call
            2: [3, 4, 8],       #escalate → create_offer or schedule_call
            3: [6, 7, 4, 8],    #create_offer → approve, reject, or schedule_call (to follow up)
            4: [3, 6, 7, 8],    #schedule_call → create_offer (revised), approve, or reject
            5: [2,  8],         #flag_manual → escalate, approve, or reject
            6: [],              #approve → terminal state (end of process)
            7: [],              #reject → terminal state (end of process)
            8: []               #cancellation_received → terminal state (end of process)
        }


        self.reward_table = {
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
        self.proba_acceptance = 0.5 #probability that the customer accepts an offer
        self.proba_response = 0.6 #probability that the customer responds to a call 
        self.proba_cancellation = 0.05 #probability that the customer cancels the application (can happen at all time steps)

        self.reset()

    def _generate_random_applicant(self):
        features = []
        for name, (low, high) in self.feature_config.items(): #static features are randomly generated at the start of the case
            if name == "loan_amount":  
                val = self.rng.choice(np.arange(low, high + 1, 5000))
           
            elif name == "credit_score":
                val = self.rng.choice(np.arange(low, high + 1, 100))
              
            elif name == "debt_to_income":
                val = self.rng.choice(np.arange(low, high + 1, 30))
          
            else:
                val = self.rng.integers(low, high + 1)
            features.append(val)
        return np.array(features, dtype=np.int32)

        

    def reset(self): #resets the environment and starts a new case
        self.static_state = self._generate_random_applicant()
        self.dynamic_state = np.zeros(self.dynamic_dim, dtype=np.int32) 
        self.last_action = 0  #start from "receive_application"
        self.state = np.concatenate([self.static_state, self.dynamic_state, [self.last_action]])
        return self.state

    def step(self, action): #depending on the selected action, this determines the transition to the next state
        assert self.action_space.contains(action)

        #If agent selected impossible action (ex: approve when no offer has been made), 
        #the case stops and the agent receives a penalty of -100
        if action not in self.get_valid_actions(): 
            return self.state, -100.0, True, False, {
                "error": f"Invalid action '{self.activity_meanings[action]}' after '{self.activity_meanings[self.last_action]}'",
                "features": dict(zip(self.feature_names, self.state.tolist())),
                "action_taken": None
            }

        dynamic_next = self.dynamic_state.copy()

        #update num_calls and num_offers and offer_amount as before
        if action == 4:  #schedule_call
            dynamic_next[0] += 1  # num_calls

            #stochastic customer response (60% chance)
            self.proba_response = 0.6
            dynamic_next[3] = 1 if self.rng.random() < self.proba_response else 0

        if action == 3:  #create_offer
            dynamic_next[1] += 1  # num_offers
            offer_amount = self.rng.choice(np.arange(1000, 20000, 10000))  # new offer amount
            dynamic_next[2] = offer_amount  # update offer_amount

            #stochastic customer acceptance (50% chance)
            self.proba_acceptance = 0.5
            dynamic_next[4] = 1 if self.rng.random() < self.proba_acceptance else 0

        #stochastic cancellation possible anytime (5% chance)
        if self.dynamic_state[5] == 0 and action not in {6,7,8}:  #only if not already cancelled and loan not approved or rejected
            self.proba_cancellation = 0.05
            dynamic_next[5] = 1 if self.rng.random() < self.proba_cancellation else 0

 
        dynamic_next = np.clip(dynamic_next, 0, np.inf).astype(np.int32)

        reward = self.reward_table.get((action), -3.0)
        
        self.dynamic_state = dynamic_next
        self.last_action = action
        self.state = np.concatenate([self.static_state, self.dynamic_state, [self.last_action]])

        done = action in {6, 7, 8}  # end if approve/reject or cancellation received
        granted = True if action in {6} else False

        return self.state, reward, done, granted, {
            "features": dict(zip(self.feature_names, self.state.tolist())),
            "action_taken": self.activity_meanings[action]
        }
    
    def get_valid_actions(self):
        valid = self.allowed_transitions.get(self.last_action, [])

     
        num_calls = self.dynamic_state[0]
        num_offers = self.dynamic_state[1]
        customer_response = self.dynamic_state[3]
        customer_accepted = self.dynamic_state[4]
        customer_cancelled = self.dynamic_state[5]
        if customer_cancelled == 1: #No action allowed after cancellation except cancellation received
            return [8] if 8 in self.allowed_transitions.get(self.last_action, []) else []

        filtered = []
        for a in valid:
            #cannot escalate if the applicant has a high credit score
            if a == 2 and self.static_state[self.feature_names.index("credit_score")] >= 600:
                continue
            #cannot flag if debt-to-income is low
            if a == 5 and self.static_state[self.feature_names.index("debt_to_income")] <= 50:
                continue
            #cannot receive cancellation if not cancelled by the customer first
            if a == 8 and customer_cancelled == 0:
                continue

            #cannot approve unless customer accepted an offer
            if a == 6 and customer_accepted == 0:
                continue
            
            if a == 6:
                cs = self.static_state[self.feature_names.index("credit_score")]
                loan = self.static_state[self.feature_names.index("loan_amount")]
                if cs <= 300 and loan > 20000:
                    continue

            #cannot create an offer if already created 3 offers
            if a == 3 and num_offers >= 3:
                continue

            #cannot schedule a call if already called 2 times
            if a == 4 and num_calls >= 4:
                continue

            #cannot schedule a call if customer already responded positively
            if a == 4 and customer_response == 1:
                continue

            #cannot approve or reject if no offer was made
            if a in {6, 7} and num_offers == 0:
                continue

            filtered.append(a)

        return filtered


    def render(self, mode="human"):
        print("\nApplicant Features:")
        for name, value in zip(self.feature_names, self.state):
            print(f"  {name:>25}: {int(value)}")
        print(f"True Creditworthiness: {'GOOD' if self.creditworthy else 'BAD'}")

env = InterpretableLoanMDP()
num_cases_generated = 30000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_FOLDER, exist_ok=True)
filename = os.path.join(DATA_FOLDER, 'df_simulation_preprocessedv2.csv')


if not os.path.exists(filename): #generate static event log
    df = generate_event_log(env, num_episodes=num_cases_generated)
    df['last_action'] = df['action']
    df['time_since_start'] = df.groupby('ID')['timestamp'].transform(lambda x: (x - x.min()).dt.days) #get the time since start of the trace
    case_outcome = df.groupby('ID')['action'].apply(lambda x: 'Success' if 'approve' in x.values else 'Failure')
    df['Outcome'] = df['ID'].map(case_outcome)
    df_success = df[df['Outcome'] == 'Success']
    df['event_number'] = df.groupby('ID').cumcount() 
    print(f"Success rate :{100*len(df_success['ID'].unique())/len(df['ID'].unique()):.2f}%")

    df.to_csv(filename, index=False)