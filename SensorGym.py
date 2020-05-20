import pandas as pd
import numpy as np
import os
import sys
import random, string
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import GPy
from timeit import default_timer as timer
from datetime import timedelta
import pickle
import multiprocessing
import yaml
import errno
class IoTNode(gym.Env):
    
    def __init__(self, **kwargs):
    
        self.mode = kwargs.get('mode','train')
        self.gamma = 0.999
        self.time_delta = 15#minutes
        self.look_ahead = int(kwargs.get('look_ahead',7)*15)#minutes
        self.deltas_ahead = int(self.look_ahead/self.time_delta)
        self.lengthscale = kwargs.get('lengthscale',np.array([0.23784978, 0.50778294, 0.11074836, 0.00024517]))
        self.var = kwargs.get('var',np.array([0.57526132, 298.13510466, 0.057]))
        self.period = kwargs.get('period',np.array([5.69027458]))


        self.gp_train = pd.date_range(start='2019-02-18 00:00:00', end='2019-02-25 00:00:00',freq='15min')
        self.rl_train = pd.date_range(start='2019-02-25 00:00:00', end='2019-03-04 00:00:00',freq='15min')
        self.rl_test = pd.date_range(start='2019-03-04 00:00:00', end='2019-03-11 00:00:00',freq='15min')
        self.df_all = pd.date_range(start='2019-02-18 00:00:00', end='2019-03-11 00:00:00',freq='15min')

        self.gp_train_iteration = 1
        self.seed_value = kwargs.get('seed',0)
        self._process_data()
        self._build_estimator()
        self.battery_max = 1.0
        self.battery_min = 0.0
        total_samples = kwargs.get('total_samples',100)
        self.energy_per_sample = (self.battery_max - self.battery_min)/total_samples
        self.min_action = kwargs.get('min_action',1)
        self.max_action = kwargs.get('max_action',96) 

        self.action_space = spaces.Discrete(self.max_action-self.min_action)

        self.low_state = np.concatenate((np.array([0.0, -1.0 ,0.0, -1.0]), -1*np.ones(self.max_action)))
        self.high_state = np.concatenate((np.array([1.0, 1.0 ,1.0, 1.0]), -1*np.ones(self.max_action)))

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset()

        
    def _get_gp_train(self):  
        df_train = self.gp_train_df
        #df_train=df_train[::10]
        # drop_indices = np.random.choice(df_train_init.index, int(len(df_train_init)*0.8), replace=False)
        # df_train = df_train_init.drop(drop_indices, axis=0)
        # df_train['time_from_last_measure']= (df_train['time']-df_train['time'].shift(periods=1)).apply(lambda x: x / np.timedelta64(int(self.time_delta),'m'))
        # df_train['time_from_last_measure']=df_train['time_from_last_measure']/self.deltas_ahead
        # df_train['last_measure']=df_train['leq'].shift(periods=1)
        # df_train = df_train[1:]
        X_train = df_train[self.input_cols].to_numpy().reshape(-1,len(self.input_cols))
        Y_train = df_train[self.output_cols].to_numpy().reshape(-1,len(self.output_cols))
        return X_train, Y_train
        
        
    def _build_estimator(self): 
        self.input_cols=['time_unit', 'is_workday','midnight_delta','time']
        self.output_cols = ['leq']

        k1= GPy.kern.Matern52(input_dim=3,ARD=True)
        k2 = GPy.kern.PeriodicExponential(input_dim=1,active_dims=[0])
        k3 = GPy.kern.White(input_dim=3)
        k = k1+k2+k3

        X_train, Y_train = np.empty(shape=(0,len(self.input_cols))),np.empty(shape=(0,len(self.output_cols)))
        np.random.seed(self.seed_value)
        for _ in range(self.gp_train_iteration):
            X, Y = self._get_gp_train()
            X_train, Y_train = np.append(X_train, X, axis=0),np.append(Y_train, Y, axis=0)
        mean_func=GPy.mappings.Constant(input_dim=3, output_dim=1, value=-0.8) 
        m= GPy.models.GPRegression(X_train[:,:-1], Y_train, kernel=k, mean_function=mean_func)#, mean_function=mean_func)#
        # m['.*lengthscale']=self.lengthscale
        # m['.*var']=self.var
        # m['.*period'][-1] = self.period
        # m[''].fix()
        m.optimize()
        self.estimator = m
    

    def _reset_estimator(self):
        X_train, Y_train = np.empty(shape=(0,len(self.input_cols))),np.empty(shape=(0,len(self.output_cols)))
        np.random.seed(self.seed_value)
        for _ in range(self.gp_train_iteration):
            X, Y = self._get_gp_train()
            X_train, Y_train = np.append(X_train, X, axis=0),np.append(Y_train, Y, axis=0)
        self.estimator.set_XY(X=X_train[:,:-1], Y=Y_train)
        
        
    def _update_estimator(self):
        new_X = np.append(self.estimator.X[self.samples_not_added:],self.selected_samples_X[-self.samples_not_added:][:,:-1], axis=0)
        new_Y  = np.append(self.estimator.Y[self.samples_not_added:],self.selected_samples_Y[-self.samples_not_added:], axis=0)
        self.estimator.set_XY(X=new_X, Y=new_Y)
        self.samples_not_added =0
        
        
    def _process_data(self, data_file='~/noise_data/noise_data.csv'):
        df = pd.read_csv(data_file)
        df['time'] = pd.to_datetime(df['time'])
        df.index=df.time
        df = df.loc[self.df_all]
        df['midnight_delta']=df.q_num/96#(48-abs(df.q_num-48))/48
        df['time_norm']=((df.index-df.index.min())/(df.index.max() - df.index.min()))
        pd.options.mode.chained_assignment = None
        df['leq_smooth']= df.leq
        for _ in range(3):
            outlier= df.leq_smooth-df.leq_smooth.shift(periods=1) >10
            outlier_idx = df.leq_smooth[outlier].index
            sub_idx =outlier_idx-pd.Timedelta(minutes=30)
            df.leq_smooth[outlier_idx] = df.leq_smooth[sub_idx]
        df['leq_smooth']=df.leq_smooth.ewm(span=5, min_periods=1).mean()
        df['is_workday'] = ((df.is_weekend==0) & (df.is_holiday == 0)).astype(int)
        df['time_unit'] =np.arange(0, len(df))/len(df)
        df['leq_orig']= df.leq
        df['leq']= df.leq_smooth
        self.leq_mean,self.leq_std = df.leq.mean(),df.leq.std()
        df.leq =(df.leq-self.leq_mean)/self.leq_std
        self.start_time,self.end_time = df.index.min(),df.index.max()
        self.gp_train_df = df.loc[self.gp_train]
        if self.mode == 'train':
            self.df = df.loc[self.rl_train]
        elif self.mode == 'test':
            self.df = df.loc[self.rl_test]
       

    def _de_normalize(self, leq):
        return (leq*self.leq_std)+self.leq_mean

    
    def _get_X_test(self, predict_ahead):
        if predict_ahead:
            df_select = self.df.loc[self.t:self.t+np.timedelta64(predict_ahead ,'m')].iloc[:-1]
        else:
            df_select = self.df.loc[self.t:].copy(deep=True)
        # last_measure =df_select.loc[self.t].leq
        # df_select['time_from_last_measure']= df_select['time'].apply(lambda x: (x - self.t)/ np.timedelta64(int(self.time_delta),'m'))
        # df_select['time_from_last_measure']=df_select['time_from_last_measure']/self.deltas_ahead
        # df_select['last_measure']=last_measure
        X_test = df_select[self.input_cols].to_numpy().reshape(-1,len(self.input_cols))
        return X_test
    
    def _kl_divergence(self,mu, var, mu_hat, var_hat):
        return 0.5*((((mu-mu_hat)**2+var)/var_hat) - 1 + np.log(var_hat/var))

    
    def _make_prediction(self,predict_ahead=None):
        new_X_test= self._get_X_test(predict_ahead)
        new_mean, Cov = self.estimator.predict_noiseless(new_X_test[:,:-1], full_cov=True)
        new_var =np.diagonal(Cov).reshape(-1, 1)
        return new_X_test,new_mean, new_var
    
    def _sample(self):
        next_sample_idx = (np.abs(self.X_test[:,-1] - np.array(self.t))).argmin()
        next_sample_X=self.X_test[next_sample_idx].reshape(-1,len(self.input_cols))
        next_sample_Y = self.df[self.output_cols].loc[self.t].to_numpy().reshape(-1,len(self.output_cols))
        self.battery -= self.energy_per_sample
        self.selected_samples_X = np.append(self.selected_samples_X, next_sample_X,axis=0)
        self.selected_samples_Y = np.append(self.selected_samples_Y, next_sample_Y,axis=0)
        kl = self._get_kl_divergence(next_sample_idx, next_sample_Y)
        self.samples_not_added +=1
        return kl

    def _get_kl_divergence(self,next_sample_idx, next_sample_Y):
        mu =next_sample_Y[0]
        var =np.expand_dims(self.var[-1], axis=0)
        mu_hat = self.mean[next_sample_idx]
        var_hat = self.var[next_sample_idx]
        # enforce that predictive variance is greater or equal to measurement noise
        var_hat = np.maximum(var, var_hat)
        kl_divergence= self._kl_divergence(mu,var, mu_hat, var_hat)
        return kl_divergence.item()

    def _get_rmse_day(self):
        start = self.t - pd.Timedelta(days=1)- pd.Timedelta(hours=self.t.hour) - pd.Timedelta(minutes=self.t.minute)
        end = start + pd.Timedelta(days=1)
        start_idx= (np.abs(self.X_test[:,-1] - np.array(start))).argmin()
        end_idx= (np.abs(self.X_test[:,-1] - np.array(end))).argmin()
        mu_hat = self.mean[start_idx:end_idx]
        mu=self.df.leq.loc[start:end].to_numpy().reshape(-1,1)
        mu = mu[:len(mu_hat)]
        RMSE = np.sqrt(((mu_hat - mu) ** 2).mean())
        return -RMSE
    
    def _get_fisher_information(self):
        start = self.t - pd.Timedelta(days=1)- pd.Timedelta(hours=self.t.hour) - pd.Timedelta(minutes=self.t.minute)
        end = start + pd.Timedelta(days=1)
        start_idx = (np.abs(self.X_test[:,-1] - np.array(start))).argmin()
        end_idx = (np.abs(self.X_test[:,-1] - np.array(end))).argmin()
    
        backward_mean, Cov = self.estimator.predict_noiseless(self.X_test[start_idx:end_idx,:-1], full_cov=True)
        backward_var =np.diagonal(Cov).reshape(-1, 1)
        fisher_information = np.sum(1/backward_var)/len(backward_var)
        self.corrected_mean = np.append(self.corrected_mean, backward_mean, axis=0)
        self.corrected_var = np.append(self.corrected_var, backward_var, axis=0)
        return fisher_information

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def _get_action_scaled(self, action):
        action = np.squeeze(action)
        return int(round(self.min_action +(action+1)*(self.max_action-self.min_action)/2))
    

    def step(self, action):
        self.last_action = action
        self.actions.append(action)
        wakeup_after = (action+self.min_action)* self.time_delta
        self.t += pd.Timedelta(minutes=int(wakeup_after))
        done = (self.t > (self.df.time.iloc[-1]-pd.Timedelta(minutes=int(self.look_ahead)))) 

        if not done:
            if (self.battery > self.battery_min):
                _ = self._sample()
                self._update_estimator()
            new_X_test,new_mean, new_var = self._make_prediction(self.look_ahead)
            X_test_t_idx = (np.abs(self.X_test[:,-1] - np.array(self.t))).argmin()
            self.pre_X_test,self.pre_mean, self.pre_var =self.X_test,self.mean,self.var

            self.X_test= np.append(self.X_test[:X_test_t_idx],new_X_test, axis =0)
            self.mean = np.append(self.mean[:X_test_t_idx],new_mean, axis=0)
            self.var = np.append(self.var[:X_test_t_idx], new_var, axis=0)

            if self.t.day != self.day:
                self.day = self.t.day
                reward = self._get_fisher_information()
            else:
                reward = 0.0      

            self.rewards.append(reward)
            self.observation =np.concatenate((np.array([self.battery,
                self.last_action/self.max_action,
                sum(new_X_test[:,1])/self.max_action,#'is_workday'
                new_X_test[0,2]]),#'midnight_delta'
                #new_mean[:self.max_action,0], 
                new_var[:self.max_action,0]))

        elif self.steps_beyond_done is None:

            new_X_test,new_mean, new_var = self._make_prediction()
            X_test_t_idx = (np.abs(self.X_test[:,-1] - np.array(self.t))).argmin()
            self.pre_X_test,self.pre_mean, self.pre_var =self.X_test,self.mean,self.var
            self.X_test= np.append(self.X_test[:X_test_t_idx],new_X_test, axis =0)
            self.mean = np.append(self.mean[:X_test_t_idx],new_mean, axis=0)
            self.var = np.append(self.var[:X_test_t_idx], new_var, axis=0)

            if self.t.day != self.day:
                start = self.t - pd.Timedelta(days=1)- pd.Timedelta(hours=self.t.hour) - pd.Timedelta(minutes=self.t.minute) 
            else:
                start = self.t - pd.Timedelta(hours=self.t.hour) - pd.Timedelta(minutes=self.t.minute)
            end = start + pd.Timedelta(days=1) #self.X_test[-1,-1]#
            start_idx = (np.abs(self.X_test[:,-1] - np.array(start))).argmin()
            end_idx = (np.abs(self.X_test[:,-1] - np.array(end))).argmin()
            backward_mean, Cov = self.estimator.predict_noiseless(self.X_test[start_idx:end_idx,:-1], full_cov=True)
            backward_var =np.diagonal(Cov).reshape(-1, 1)
            fisher_information = np.sum(1/backward_var)/len(backward_var)
            self.corrected_mean = np.append(self.corrected_mean, backward_mean, axis=0)
            self.corrected_var = np.append(self.corrected_var, backward_var, axis=0)
            self.steps_beyond_done = 0
            reward = fisher_information 
            self.rewards.append(reward)
            print(start, end)

            mu_hat = self.corrected_mean
            var_hat = self.corrected_var
            mu=self.df.leq[:len(mu_hat)].to_numpy().reshape(-1,1)
            var=np.expand_dims(self.var[-1], axis=0)
            overall_kl =self._kl_divergence(mu,var,mu_hat,var_hat).sum()
            RMSE = np.sqrt(((mu_hat - mu) ** 2).mean())

            print("overall_kl: {:.2f}, RMSE: {:.2f}".format(overall_kl, RMSE))
            print("overall_rewards: {:.2f}, mean_action: {:.2f}".format(sum(self.rewards), sum(self.actions)/len(self.actions)))
            
        else:
            self.steps_beyond_done += 1
            reward = 0.0
            
        return self.observation, reward, done, {}
        

    def reset(self):
        self.actions = []
        self.rewards = []
        self.last_action= -1
        self.t = self.df.time.iloc[0]
        self.day = self.t.day
        self.battery = self.battery_max 
        self._reset_estimator()
        self.X_test,self.mean, self.var=self._make_prediction(self.look_ahead)
        self.pre_X_test,self.pre_mean, self.pre_var =self.X_test,self.mean,self.var
        self.corrected_mean = np.empty(shape=(0,1))
        self.corrected_var = np.empty(shape=(0,1))

        self.steps_beyond_done = None
        self.selected_samples_X = np.empty(shape=(0,len(self.input_cols)))
        self.selected_samples_Y = np.empty(shape=(0,len(self.output_cols)))
        self.samples_not_added = 0
        self.observation =np.concatenate((np.array([self.battery,
            self.last_action/self.max_action,
            sum(self.X_test[:,1])/self.max_action,
            self.X_test[0,2]]),
            #self.mean[:self.max_action,0], 
            self.var[:self.max_action,0]))

        return self.observation
        

    def render(self, mode='human', **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns
        save_name = kwargs.get('save_name','default_name')
        episode_rewards = kwargs.get('episode_rewards',None)
        xlim = kwargs.get('xlim',None)
        mean = self._de_normalize(self.mean)
        var = self.var*self.leq_std**2

        corrected_mean = self._de_normalize(self.corrected_mean)
        corrected_mean = np.append(corrected_mean,mean[len(corrected_mean):], axis=0)

        corrected_var= self.corrected_var*self.leq_std**2
        corrected_var = np.append(corrected_var,var[len(corrected_var):], axis=0)

        n_std_devs = 3
        
        pd.plotting.register_matplotlib_converters()

        sns.set_style("darkgrid")
        sns.set_context("notebook", font_scale=1.5, 
                        rc={"lines.linewidth": 1.5,"font.zise":12, 
                            "text.color":"black","font.weight":'bold'})
        fig,  ax2 = plt.subplots(1,  figsize=(20,4), gridspec_kw = {'wspace':0, 'hspace':0.0})

        if episode_rewards != None:
            fig.suptitle('Episode Reward = %.3f '%episode_rewards, y=0.93,fontsize=14)

        # ax1.plot(self.df.index, self._de_normalize(self.df.leq), label='True values')
        # # ax1.axhspan(58, 75, facecolor='r', alpha=0.2, zorder=-100)
        # # ax1.axhspan(47, 58, facecolor='orange', alpha=0.2, zorder=-100)
        # # ax1.axhspan(30, 47, facecolor='g', alpha=0.2, zorder=-100)
        # ax1.scatter(self.selected_samples_X[:,-1],
        #             self._de_normalize(self.selected_samples_Y),
        #             color='k',marker='x',s=500, label='selected samples $Y$')
        # ax1.plot(self.X_test[:,-1], mean, 'r', lw=2, label='predicted mean')
        # ax1.fill_between(self.X_test[:,-1],
        #                  mean[:,0]-n_std_devs*np.sqrt(var[:,0]), 
        #                  mean[:,0]+n_std_devs*np.sqrt(var[:,0]),
        #                  color='C0', alpha=0.2, label= 'Prediction Interval ($\pm %g\hat{\sigma}$)'%n_std_devs)
        # ax1.set_ylabel('Leq')
        # ax1.axvline(self.t,color='y', linestyle='--',  alpha=0.5)
        # ax1.set_xlim(self.X_test[0,-1],self.X_test[-1,-1])
        # ax1.set_ylim(35, 70)
        #ax1.legend(loc=0)


        ax2.plot(self.df.index, self._de_normalize(self.df.leq), label='True values')
        # ax2.axhspan(58, 75, facecolor='r', alpha=0.2, zorder=-100)
        # ax2.axhspan(47, 58, facecolor='orange', alpha=0.2, zorder=-100)
        # ax2.axhspan(30, 47, facecolor='g', alpha=0.2, zorder=-100)
        ax2.scatter(self.selected_samples_X[:,-1],
                    self._de_normalize(self.selected_samples_Y),
                    color='k',marker='x',s=500, label='selected samples $Y$')
        ax2.plot(self.X_test[:,-1], corrected_mean, 'r', lw=2, label='corrected mean')
        ax2.fill_between(self.X_test[:,-1],
                         corrected_mean[:,0]-n_std_devs*np.sqrt(corrected_var[:,0]), 
                         corrected_mean[:,0]+n_std_devs*np.sqrt(corrected_var[:,0]),
                         color='C0', alpha=0.2, label= 'Prediction Interval ($\pm %g\hat{\sigma}$)'%n_std_devs)
        ax2.set_ylabel('Leq')
        ax2.axvline(self.t,color='y', linestyle='--',  alpha=0.5)
        if xlim != None:
            ax2.set_xlim(xlim[0],xlim[1])
        else:
            ax2.set_xlim(self.X_test[0,-1],self.X_test[-1,-1])
        ax2.set_ylim(40, 65)
        ax2.set_xlabel('Time')
        #ax2.legend(loc=0)

        dir_name = 'figures/'
        os.makedirs(dir_name, exist_ok=True)
        if save_name != 'default_name':
            print('saving_fig')
            plt.savefig(os.path.join(dir_name, save_name +".pdf"),  bbox_inches='tight')
        plt.show()
    
    def _scale_down_action(self, action):
        return np.array([2*(action-self.min_action)/(self.max_action-self.min_action)-1])

    def close(self):
        pass
    

