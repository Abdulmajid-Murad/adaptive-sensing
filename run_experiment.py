
from SensorGym import IoTNode
import pandas as pd
import numpy as np
import os
import sys
import random, string
import math
from timeit import default_timer as timer
from datetime import timedelta
import pickle
import multiprocessing
import yaml
import errno

def unfold_configurations(config, id_prefix='A'):
    from itertools import product
    v = []
    for values in config.values():
        v.append(values if type(values) == list else [values])
    values = list(product(*v))
    configurations = []
    for idx, value_set in enumerate(values):
        c = dict(zip(config.keys(), value_set))
        c['id'] = '{}{:02d}'.format(id_prefix,idx)
        configurations.append(c)
    return configurations

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def build_env(log_dir,env_kwargs, nenv=None):
    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines.common import set_global_seeds
    from stable_baselines.bench import Monitor
    import multiprocessing

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = nenv or ncpu
    
    def make_env(rank,seed=0): # pylint: disable=C0111
        def _thunk():
            env =IoTNode(**env_kwargs)
            env.seed(seed+rank)
            env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
            return env
        set_global_seeds(0)
        return _thunk
    if nenv > 1: 
        VecEnv=SubprocVecEnv([make_env(i) for i in range(nenv)])
    else: 
        VecEnv=DummyVecEnv([make_env(0)])
    return VecEnv #VecNormalize(VecEnv, norm_obs=True, norm_reward=True)


def generate_trial_path(study_path):
    while True:
        id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) 
        path = os.path.join(study_path, id)
        if not os.path.exists(path):
            return path, id


def run_process(study_name,alg_param, env_param, log_path='.'):
    study_path = os.path.join(log_path, study_name)
    make_sure_path_exists(study_path)
    trial_path, trial_id = generate_trial_path(study_path)
    make_sure_path_exists(trial_path)


    with open(trial_path+ '/alg_param.pkl', "wb+") as outfile:
        pickle.dump(alg_param, outfile)

    with open(trial_path+ '/env_param.pkl', "wb+") as outfile:
        pickle.dump(env_param, outfile)


    num_nodes = alg_param['num_nodes']
    num_layers = alg_param['num_layers']
    learning_rate=alg_param['learning_rate']
    alg = alg_param['alg']
    nenv = alg_param['nenv']
    env = build_env(trial_path,env_param, nenv=nenv)

    if alg == 'dqn':
        from stable_baselines.deepq.policies import MlpPolicy
        from stable_baselines import DQN
        call_iter = 1000
        policy_kwargs = dict(layers=[num_nodes for _ in range(num_layers)])
        model = DQN(MlpPolicy, env, 
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=trial_path)
    #DDPG calls back every step of every rollout
    elif alg == 'ddpg':
        from stable_baselines.ddpg.policies import MlpPolicy
        from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
        from stable_baselines import DDPG
        call_iter = 1000
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        policy_kwargs = dict(layers=[num_nodes for _ in range(num_layers)])
        model = DDPG(MlpPolicy, env,
                    verbose=1, 
                    param_noise=param_noise, 
                    action_noise=action_noise,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=trial_path)

    elif alg == 'td3':
        from stable_baselines import TD3
        from stable_baselines.td3.policies import MlpPolicy
        from stable_baselines.common.vec_env import DummyVecEnv
        from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
        call_iter = 1000
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        policy_kwargs = dict(layers=[num_nodes for _ in range(num_layers)])
        model = TD3(MlpPolicy, env,
                    verbose=1,
                    action_noise=action_noise,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=trial_path)

    #PPO1 calls back only after every rollout
    elif alg == 'ppo2':
        from stable_baselines.common.policies import MlpPolicy
        from stable_baselines import PPO2
        call_iter = 100
        policy_kwargs = dict(net_arch=[num_nodes for _ in range(num_layers)])
        model = PPO2(MlpPolicy,env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=learning_rate,
                tensorboard_log=trial_path,
                n_steps=alg_param['n_steps'],
                noptepochs=alg_param['noptepochs'],
                nminibatches=alg_param['nminibatches'],
                gamma=alg_param['gamma'],
                ent_coef=alg_param['ent_coef'],
                cliprange=alg_param['cliprange'],
                lam=alg_param['lam'])



    best_mean_reward, n_steps = -np.inf, 0
    #callback frequency differs among algorithms
    def callback(_locals, _globals):
        from stable_baselines.results_plotter import load_results, ts2xy
        nonlocal n_steps, best_mean_reward, call_iter
        # Print stats every 1000 call
        if (n_steps + 1) % call_iter  == 0:
          # Evaluate policy training performance
            x, y = ts2xy(load_results(trial_path), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-200:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(trial_path + '/best_model.pkl')
        n_steps += 1
        return True

    # model= DDPG.load('log/A00/best_model.pkl')
    # model.set_env(env)
    print(f"Starting to train {trial_id}")
    model.learn(total_timesteps=int(1e6),
                tb_log_name='tb_log',
                callback=callback)

    model.save(trial_path + '/fully_trained_model')




if __name__ == "__main__": 

    alg_args  = {
        'num_layers': [2,3,4],
        'num_nodes': [32,64],
        'n_steps': [16,32,128,256,2048],
        'noptepochs': [4,10,20],
        'nminibatches': [1,4,8,32],
        'gamma':[0.99, 0.999],
        'ent_coef':[0.0, 0.01],
        'cliprange': [0.2],
        'lam': [0.98, 0.95],
        'learning_rate':list(np.logspace(-1, -4, num=100)),
        'nenv':[8,16],
        'alg':['ppo2']
        }
    alg_param = {key: random.sample(value, 1)[0] for key, value in alg_args .items()}
    env_param = {
        'mode':'train',
        'look_ahead': 12,
        'min_action': 1,
        'max_action':12,
        'total_samples':100,
        'lengthscale': np.array([0.1, 0.01,0.1,0.001]), 
        'var':np.array([1.0,1.0,0.001]), 
        'period':np.array([0.14]),
        'seed':0
    }
    study_name= 'run_v10'
            
    run_process(study_name,alg_param, env_param, log_path='.')


