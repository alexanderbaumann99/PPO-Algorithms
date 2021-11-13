import torch
from utils import *
import copy
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
     
        # Define actor network
        self.actor=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,self.action_space.n),
            nn.Softmax(dim=-1)
        )
        
        # Define critic network
        self.critic=nn.Sequential(
            nn.Linear(env.observation_space.shape[0],64),
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
        
        #Define learning rates and optimizers
        self.lr_a=2e-4
        self.lr_c=2e-4
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),self.lr_a)
        self.optimizer_critic= torch.optim.Adam(self.critic.parameters(),self.lr_c)

        # Define algorithm variables
        self.clip=opt.clip
        self.ppo=opt.ppo
        self.dl=opt.dl
        
        #Define hyperparameters
        self.K=opt.K_epochs
        self.discount=0.99          # Discount factor
        self.gae_lambda=0.95        # Lambda of TD(lambda) advantage estimation
        
        #Hyperparameters of clipped PPO
        self.eps_clip=0.2
        # Hyperparameters of KL-Div Algo
        self.beta=1.
        self.delta=0.01    
        
        #Initialize memory
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.new_states=[]
        self.values=[]
        
        #counters
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

        if self.clip:
            self.learn_clip()
        elif self.dl:
            self.learn_kl()
        elif self.ppo:
            self.learn_ppo()
        


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
