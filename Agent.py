import argparse
import sys
import matplotlib
from numpy import dtype, log
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
import gym
import gridworld
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import copy
from torch.distributions import Categorical
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


#Xavier weight initialization
def init_weights(m):

    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(m.bias)



class Agent(object):
  
    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.test=False
        self.nbEvents=0
   
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.device=torch.device('cpu') 
        self.actor=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,self.action_space.n),
            nn.Softmax(dim=-1)
        )
        self.critic=nn.Sequential(
            nn.Linear(self.ob_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.actor_old=copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.lr_a=2e-4
        self.lr_c=2e-4
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),self.lr_a)
        self.optimizer_critic= torch.optim.Adam(self.critic.parameters(),self.lr_c)

        self.clip=opt.clip

        self.K=opt.K_epochs
        self.beta=1.
        self.delta=0.01
        self.eps_clip=0.2
        self.discount=0.99
        self.gae_lambda=0.95

        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

        self.actor_count=0
        self.critic_count=0
       
    
    
    def act(self, obs):
        
        #Calculate distribution of policy
        prob=self.actor(torch.FloatTensor(obs).to(self.device))
        dist=Categorical(prob)
        #sample action w.r.t policy
        action=dist.sample()
        
        #store values
        if not self.test:
            self.log_probs.append(dist.log_prob(action))
            self.actions.append(action.detach())
            self.states.append(torch.FloatTensor(obs).to(self.device))
            self.values.append(self.critic(torch.FloatTensor(obs).to(self.device)).detach())
       

        return action.item()

    
    #learning algorithm of PPO with Adaptive Kullback-Leibler divergence 
    def learn_kl(self):
      
        #Compute the TD(lambda) advantage estimation
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()
        rewards = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*rewards[t+1]
                delta=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()

            adv=adv*self.discount*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv

        
                
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        #Normalize the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)
        
        
        pi_old=self.actor(old_states).view((-1,self.action_space.n))

        #state_values=self.critic(old_states).view((-1,self.action_space.n))        
        #state_value=state_values.gather(1,old_actions.view((-1,1)))
        
        state_value=self.critic(old_states).view(-1) 
        
        for _ in range(self.K):
            
            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)
            ratios=torch.exp(log_probs-old_logprobs.detach())
            
            #PPO Loss
            loss1=torch.mean(ratios*advantage.detach())
            #KL-Divergence Loss
            loss2=F.kl_div(input=probs,target=pi_old.detach(),reduction='batchmean')
            
            #Actor update
            actor_loss=- (loss1-self.beta*loss2)
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        #KL-Divergence update
        DL=F.kl_div(input=probs.view((-1,self.action_space.n)),target=pi_old.view((-1,self.action_space.n)),reduction='batchmean')
        if DL>=1.5*self.delta:
            self.beta*=2
        if DL<=self.delta/1.5:
            self.beta*=0.5

        #Critic update
        loss=F.smooth_l1_loss(rewards,state_value.view(-1))
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()
        
        #Clear memory
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
       
    #learning algorithm of PPO
    def learn_ppo(self):

        #Compute the TD(lambda) advantage estimation
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()
        rewards = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                #td_error = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*rewards[t+1]
                #td_error=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1]-self.values[t]
                
            advantage[t]=rewards[t]-self.values[t]

        
                
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        #Normalize the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)
        
        pi_old=self.actor(old_states).view((-1,self.action_space.n))

        #state_values=self.critic(old_states).view((-1,self.action_space.n))        
        #state_value=state_values.gather(1,old_actions.view((-1,1)))
        
        state_value=self.critic(old_states).view(-1) 
        
        for _ in range(self.K):

            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)
            ratios=torch.exp(log_probs-old_logprobs.detach())
            
            #Use only the PPO Loss here
            loss1=torch.mean(ratios*advantage.detach())
            actor_loss=-loss1
            
            #Actor update
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
        
        #Critic update
        loss=F.smooth_l1_loss(rewards,state_value.view(-1))
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        #Clear memory
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
        
        
    #learning algorithm of PPO with clipped objective
    def learn_clip(self):

        #Compute TD(lambda) advantage estimation
        last_val=self.critic(torch.FloatTensor(self.new_states[-1]).to(self.device)).item()
        rewards = np.zeros_like(self.rewards)
        advantage = np.zeros_like(self.rewards)
        adv=0.
        for t in reversed(range(len(self.rewards))):
            if t==len(self.rewards)-1:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*last_val
                delta = self.rewards[t]+self.discount*(1-self.dones[t])*last_val - self.values[t].item()
            else:
                rewards[t]=self.rewards[t]+self.discount*(1-self.dones[t])*rewards[t+1]
                delta=self.rewards[t]+self.discount*(1-self.dones[t])*self.values[t+1].item()-self.values[t].item()

            adv=adv*self.discount*self.gae_lambda*(1-self.dones[t])+delta
            advantage[t]=adv
          
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        advantage = torch.FloatTensor(advantage).to(self.device)
        #Normalize the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
       
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.log_probs, dim=0)).detach().to(self.device)

        state_values=self.critic(old_states).view(-1)       
        
        
        for _ in range(self.K):

            probs = self.actor(old_states)
            dist=Categorical(probs)
            log_probs=dist.log_prob(old_actions)
            ratios=torch.exp(log_probs-old_logprobs.detach())
            
            #PPO-Loss
            loss1=ratios*advantage.detach()
            #Clipped Loss
            loss2=torch.clamp(ratios,min=1-self.eps_clip,max=1+self.eps_clip)*advantage.detach()
            
            #Actor update
            actor_loss= -torch.mean(torch.min(loss1,loss2))         
            self.actor_count+=1
            self.actor_loss=actor_loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

        #Critic update
        loss=F.smooth_l1_loss(rewards,state_values)
        self.critic_loss=loss
        self.critic_count+=1
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        #Clear memory
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]

    def learn(self):

        if self.clip==True:
            self.learn_clip()
        else:
            self.learn_kl()
        


    def store(self,ob, action, new_obs, reward, done, it):
       
        if not self.test:

            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.new_states.append(new_obs)
 

   #defines the timesteps when the agent learns
    def timeToLearn(self,done):
        if self.test:
            return False
    
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "RandomAgent")
    
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = Agent(env,config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
      
        ob = env.reset()

        # Check if verbose
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # Check if it is a testing episode
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

       # End of testing, evaluate testing results and go back to train modus
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        j = 0
        if verbose:
            env.render()

        new_obs = agent.featureExtractor.getFeatures(ob)
        
        while True:
            if verbose:
                env.render()

            ob = new_obs
            
            action= agent.act(ob)
            new_obs, reward, done, _ = env.step(action)      
            agent.store(ob, action, new_obs, reward, done,j)

            j+=1

            # If we reached the maximal length per episode
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            
            rsum += reward
            
            # If it is time to learn, let the agent learn
            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("actor loss", agent.actor_loss, agent.actor_count)
                logger.direct_write("critic loss", agent.critic_loss, agent.critic_count)
                
            # If episode is done, evaluate the results of this episode and start a new episode
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                #agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break
                
      
    env.close()
