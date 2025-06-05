python preprocessing.py
python simulated_online_nopretraining.py 

python simulated_offline_online_RL.py --state_abstraction=False
python simulated_offline_online_RL.py --state_abstraction=structural
python simulated_offline_online_RL.py --state_abstraction=k_means --k=10
python simulated_offline_online_RL.py --state_abstraction=k_means --k=50
python simulated_offline_online_RL.py --state_abstraction=k_means --k=200
python simulated_offline_online_RL.py --state_abstraction=k_means_features --k=10
python simulated_offline_online_RL.py --state_abstraction=k_means_features --k=50
python simulated_offline_online_RL.py --state_abstraction=k_means_features --k=200
python simulated_offline_online_RL.py --state_abstraction=last_action


python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=False 
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=structural
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means --k=10
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means --k=50
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means --k=200
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means_features --k=10
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means_features --k=50
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=k_means_features --k=200
python process_autopilot_training.py --dataset=bpi2017 --state_abstraction=last_action

python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=False
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=structural
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means --k=10
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means --k=50
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means --k=200
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means_features --k=10
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means_features --k=50
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=k_means_features --k=200
python process_autopilot_training.py --dataset=bpi2012 --state_abstraction=last_action
