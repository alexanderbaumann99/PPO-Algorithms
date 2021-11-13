from utils import *
from Agent import Agent
import numpy as np


if __name__ == '__main__':
    env, config, outdir, logger = init('config.yaml')
    
    freqTest = config["freqTest"]
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

        new_obs=ob
        
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
                mean += rsum
                rsum = 0

                break
                
      
    env.close()
