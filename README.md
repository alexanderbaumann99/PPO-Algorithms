# PPO-Algorithm

<p>I implemented three versions of the PPO-Algorithm as proposed in John Schulman et al. 'Proximal policy optimization algorithms' (https://arxiv.org/abs/1707.06347). </p>

<ul>
	<li> PPO without clipping or penalty <br/>color: red </li>
	<li> PPO with clipped objective <br/>color: orange </li>
	<li> PPO with adaptive Kullback-Leibler penalty<br/>color: blue </li>
</ul>


We test these three versions on the 'CartPole-v1' environment. <p>We see that the PPO with adpative KL-penalty outperforms the other two algorithms in this example. However, the second plot shows that this alogrithm takes the longest on the other hand , but still outperforms on a relative basis.<br/> PPO with adpative KL-Divergence outperforms also while testing.</p>
<p> Note that the first two plots are smoothed.</p>


### Reward per episode:
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/TrainReward_per_Episode.svg?raw=true)
### Relative reward to the time:
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/relative_TrainReward.svg?raw=true)

### Reward per test episode:
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/TestReward.svg?raw=true)




  
