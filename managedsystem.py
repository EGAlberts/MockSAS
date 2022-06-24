from email import generator
from masced_bandits.bandits import init_bandit
from masced_bandits.bandit_options import initialize_arguments
from environmentgrammar import environment_grammar, EnvironmentTransformer
from lark import Lark
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pprint 

def notfoo(something):
    return "notfoo"

class MockSAS:
    def __init__(self, num_sys, policies, parse_tree):
        #could be multiple systems with one environment
        self.managing_systems = []
        for i in range(num_sys):
            
            transformer_dict = EnvironmentTransformer().transform(parse_tree)
            managed = [self.ManagedSystem(transformer_dict["reward_generator"], transformer_dict["feature_generator"])] #extend this with multiple managed systems if thats what you want to model.
            EnvironmentTransformer.environment_grabber = managed[0].get_observations
            
            m_sys = self.ManagingSystem(policies[i], managed, list(transformer_dict["all_arms"]))
            self.managing_systems.append(m_sys) 


    def operation(self, res = {}):
        for m_sys in self.managing_systems:
            managed_busy = [True]
            
            while all(managed_busy):
                managed_busy = []
                for managed in m_sys.managed:
                    acks = managed.notify_observers()
                    if(managed.environment):
                        managed.environment.notify_observers()
                    #print("acks:" + str(acks))
                    busy = all(acks)
                    #print("busy: " + str(busy))
                    managed_busy.append(busy)
            res[m_sys.name] = m_sys.avg_rw_record
        
        return res
            
            # if(input("> ") == 'stop'): break

    class ManagingSystem:
        def __init__(self, policy_tuple, managed, arms):
            self.name = str(policy_tuple) + "_msys"           
            initialize_arguments(arms, 0, bounds=(0,1))
            self.policy = init_bandit(**policy_tuple)
            self.current_action = arms[0]
            self.managed = managed
            self.round = 1
            self.average_reward = 0
            self.avg_rw_record = []

            for managed_system in managed:
                managed_system.register_observer(self)

        def notify(self, environment):
            #metrics = environment.metrics; reward_function(metrics)
            if(environment):
                reward = next(environment[self.current_action])
                #add check for none reward due to inactive arm.
                #print("received " + str(reward))
                self.current_action = self.policy.start_strategy(reward)
                self.average_reward = self.average_reward + ((1/self.round) * (reward - self.average_reward))
                self.avg_rw_record.append(self.average_reward)

                return True
            else:
                #end of trace
                #print("trace ended")
                return False


    """
    This is the environment with which the SAS interacts.
    It should in essence be a mapping from actions to rewards.
    """
    class ManagedSystem:
        def __init__(self, generator, env_generator):
            self._observers = []
            self.observations = {}
            self.generator = generator

            if(env_generator): #Essentially, if there are env variables to observe
                self.environment = self.Environment(env_generator)
                self.environment.register_observer(self)

        def get_observations(self):
            return self.observations
        def notify(self, new_observations):
            self.observations = new_observations
          
        def register_observer(self, observer):
            self._observers.append(observer)
        
        def notify_observers(self):
            round_dists = next(self.generator)
            acks = []
            for obs in self._observers:
                acks.append(obs.notify(round_dists))
            return acks

        class Environment:
            def __init__(self, generator):
                self._observers = []
                self.generator = generator
            def register_observer(self, observer):
                self._observers.append(observer)
        
            def notify_observers(self):
                round_dists = next(self.generator)
                acks = []
                for obs in self._observers:
                    acks.append(obs.notify(round_dists))
                return acks



parser = Lark(environment_grammar)
mission_statement = "ComplexSystem.txt"
try:
    with open(mission_statement, 'r') as source:
        source_code = source.read()
except IOError:
    print('Something is wrong with the source code file')

try:
    parse_tree = parser.parse(source_code)
    #print(parse_tree.pretty())
    
    
except Exception as e:
    print("Syntax error, details below: \n")
    print(e)
    exit(1)

final_res = {}

configs = [[{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "egreedy", 'epsilon': "0.2"}],[{'name': "egreedy", 'epsilon': "0.3"}],[{'name': "egreedy", 'epsilon': "0.4"}],[{'name': "egreedy", 'epsilon': "0.5"}], [{'name': "egreedy", 'epsilon': "0.6"}],[{'name': "egreedy", 'epsilon': "0.7"}],[{'name': "egreedy", 'epsilon': "0.8"}],[{'name': "egreedy", 'epsilon': "0.9"}],[{'name': "egreedy", 'epsilon': "1.0"}]]

for config in configs:
    final_res[str( str(config[0]['name']) + str(config[0].get('formula',""))  + str(config[0].get('epsilon',"")) + str(config[0].get('decay_rate',"")))] = []
for i in range(1):
    all_res = {}
    for config in configs:
        #input(str(config) + ">")
        np.random.seed(i * 1337)
        mksas = MockSAS(1,config, parse_tree)
        res = mksas.operation({})
        for key in res.keys():
            #input(str(key) + ">>")
            #input(str(res[key][-1]) + ">>>")
            final_res[str(str(config[0]['name']) + str(config[0].get('formula',""))  + str(config[0].get('epsilon',"")) + str(config[0].get('decay_rate',"")))].append((res[key][-1]))
            
pprint.pprint(final_res)
print(str(final_res.keys()))

plt.boxplot(final_res.values())
plt.xticks(ticks=list(range(1,len(configs)+1,1)),labels=final_res.keys())

#plt.xticks(final_res.keys())
#plt.legend()
plt.show()  