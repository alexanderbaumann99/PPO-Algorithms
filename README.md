# PPO-Algorithm

<p>I implemented three versions of the PPO-Algorithm:</p>

<ul>
	<li> <p>Usual PPO as in ... </p><p>color: red</p> </li>
	<li> <p>PPO with clipped objective as in ... <p>color: bright blue</p> </li>
	<li> <p>PPO with adaptive Kullback-Leibler Divergence as in ... <p>color: dark blue</p> </li>
</ul>


We test these three versions on the 'CartPole-v1'environment.

Train:
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/TrainReward_per_Episode.svg?raw=true)
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/relative_TrainReward.svg?raw=true)

We see that the PPO with adpativ KL-Divergence outperforms the other two algorithms in this example. However, we also see in second plot that this alogrithm takes the longest time.

Test:
![alt text](https://github.com/alexbaumi/PPO-Algorithm/blob/main/figures/TestReward.svg?raw=true)

PPO with adpativ KL-Divergence outperforms also while testing.


  
