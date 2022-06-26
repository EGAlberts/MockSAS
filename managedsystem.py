from masced_bandits.bandits import init_bandit
from masced_bandits.bandit_options import initialize_arguments
from environmentgrammar import environment_grammar, EnvironmentTransformer
from lark import Lark
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pprint 
from math import floor

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
                    try:
                        managed.environment.notify_observers()
                    except AttributeError:
                        #no environment'
                        pass
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
