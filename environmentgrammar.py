from lark import Transformer
import sys
from sys import exit
from types import GeneratorType
import numpy as np
from scipy.stats import truncnorm
import userfunctions
import code
import pprint
import userfunctions 
environment_grammar = """
    ?start: INT? context_definition context_trace
    
    context_trace: "Trace:" (indefinite~1 | contextpair+ indefinite?)
    context_definition: context+
    indefinite: "(" CNAME ")"
    contextpair: "(" CNAME "," LENGTH ")"
    context: CNAME "{" environment_definition? arm_definition "}"
    arm_definition: "arms: {" actionpair+ "}"
    environment_definition:  "features: {" environpair+ "}" 
    environpair:  CNAME ":" (NUMBER | dist) 
    actionpair: CNAME ":" dist
    dist: (uniform | normal | truncated_normal | logisitic | inactive | constant | userfunction)
    normal: "normal(" mean "," stdev ")"
    uniform:  "uniform(" lower "," upper ")"
    logisitic: "logistic(" mean "," stdev ")"  
    truncated_normal: "truncnorm(" lower "," upper "," mean "," stdev ")"
    constant: "constant(" value ")"
    inactive: "inactive()"
    userfunction: CNAME "(" value ("," value)* ")"

    value: (NUMBER | VARIABLE | userfunction)

    VARIABLE: CNAME
    stdev: value
    upper: value
    lower: value
    mean: value
    LENGTH: INT

    %import common.INT
    %import common.NUMBER
    %import common.CNAME
    %import common.WORD
    %import common.WS
    %ignore WS
"""

class EnvironmentTransformer(Transformer):
    all_arms = set()

    def generator_factory(self, args):
        funcname, *funpara = args
        
        try:
            my_function = getattr(userfunctions, funcname)
        except AttributeError:
            raise RuntimeError("Used a user-defined function which doesn't exist")

        def generic_generator(variables = None):
            if(variables):
                while True:
                    try:
                        env_state = self.environment_grabber()
                    except AttributeError:
                        env_state = variables
                    final_params = []
                    for para in funpara:
                        if(type(para) == str):
                            variable_value = env_state[para]
                            if(isinstance(variable_value,GeneratorType)):
                                final_params.append(next(variable_value))
                            elif(callable(variable_value)):
                                raise RuntimeError('Nested more than implemented')
                            else:
                                final_params.append(variable_value)
                        elif(isinstance(para, GeneratorType)):
                            final_params.append(next(para))
                        elif(callable(para)):
                            gen_from_callable = para(variables)
                            x = next(gen_from_callable)
                            final_params.append(x)
                            funpara[funpara.index(para)] = gen_from_callable #should now remain a generator for next time

                        else:
                            final_params.append(para)

                    yield my_function(*final_params)
                    
            else:
                while True:
                    final_params = []
                    for para in funpara:
                        if(isinstance(para, GeneratorType)):
                            final_params.append(next(para))
                        else:
                            final_params.append(para)
                    yield my_function(*final_params)

        if(any(type(param) == str for param in funpara) or any(callable(param) for param in funpara)):
            return generic_generator
        else:
            return generic_generator()

    def start(self,arg):

        return_dict = {}

        if(len(arg) == 3):
            given_seed, context_and_feature_dict, context_tuples = arg
            #np.random.seed(given_seed)
            print("seed set")
        else:
            context_and_feature_dict, context_tuples = arg
        
        context_to_arm, context_to_feature = context_and_feature_dict

        #add the indefinite functionality
        
        for c_t in context_tuples: 
            if(c_t[0] != "indefinite"):
                if(c_t[0] not in context_to_arm): raise RuntimeError("Context specified in trace which was not defined")

        def reward_generator():
            rounds_elapsed = 0
            context_index = 0
            end_of_trace = False
            while True:
                if(end_of_trace): yield None
                current_context = context_tuples[context_index][0]
                
                yield context_to_arm[current_context]
                rounds_elapsed+=1
                if(rounds_elapsed > context_tuples[context_index][1]): 
                    context_index+=1
                    if(context_index >= len(context_tuples)):
                        #trace over
                        end_of_trace = True
                    rounds_elapsed = 0 #restart counter for new context
        
        def feature_generator():
            rounds_elapsed = 0
            context_index = 0
            end_of_trace = False
            while True:
                if(end_of_trace): yield None
                current_context = context_tuples[context_index][0]
                yield context_to_feature[current_context]
                rounds_elapsed+=1
                if(rounds_elapsed > context_tuples[context_index][1]): 
                    context_index+=1
                    if(context_index >= len(context_tuples)):
                        #trace over
                        end_of_trace = True
                    rounds_elapsed = 0 #restart counter for new context
        r_gen = reward_generator()
        f_gen = None
        if(context_to_feature): f_gen = feature_generator()
        self.all_arms = list(self.all_arms)
        self.all_arms.sort()
        return_dict["all_arms"] = self.all_arms
        return_dict["reward_generator"] = r_gen
        return_dict["feature_generator"] = f_gen

        return return_dict

    def parse_dictionary(self, pair_list):
        new_dictionary = {}
        for entry in pair_list:
            entry_key, entry_value = entry
            if(entry_key in new_dictionary): raise RuntimeError("Error while parsing dictionary: Same key specified multiple times, names not unique")
            new_dictionary[entry_key] = entry_value 
        
        return new_dictionary


    def context_definition(self,contexts):
        context_to_feature = {}

       

        for context in contexts:
            feature_dictionary = context.pop()
            if(feature_dictionary):
                context_to_feature[context[0]] = feature_dictionary
                
            
        context_dict = self.parse_dictionary(contexts)

        return context_dict,context_to_feature
    def arm_definition(self, args):
        return args

    def environment_definition(self, args):
        # variable_dict = {}
        # for var in args:
        #     var_name, var_value = var
        #     if(var_name in variable_dict): raise RuntimeError("Same variable specified multiple times: variable names not unique")
 
        #     variable_dict[var_name] = var_value

        # return variable_dict
        return self.parse_dictionary(args)
    def context_trace(self,arg):
        for i, context_tuple in enumerate(arg):
            if(len(context_tuple) == 1):
                arg[i] = context_tuple + (sys.maxsize,) #this makes indefinite last as long as max int size.
        return arg

    def context(self,argss):
        # context_dict = {}
        action_dict = {}
        variable_dict = None
        if(len(argss) == 3):
            context_name, variable_dict, action_pairs = argss
        else: 
            context_name, action_pairs = argss

        for action_pair in action_pairs:
            action_name, action_gen = action_pair

            if(callable(action_gen)): action_gen = action_gen(variable_dict) #if it is not a generator yet that implies it needs to use the variables.
            self.all_arms.add(action_name)
            if(action_name in action_dict): raise RuntimeError("Same action specified multiple times: action names not unique within context")
            action_dict[action_name] = action_gen
        
        #code.interact(local=locals())
        return [context_name, action_dict, variable_dict]
    def actionpair(self,arg): return arg
    def environpair(self,arg): return arg
    def dist(self,args): return args[0]

    # def environment_grabber(self):
    #     return "foo"
    
    def normal(self,argg):
        args = ["normal"] + argg
        genn = self.generator_factory(args)

        return genn

    def uniform(self,argg):
        args = ["uniform"] + argg
        genn = self.generator_factory(args)

        return genn
        # def normal_generator(variables = None):
        #     if(variables):
        #         while True:
        #             try:
        #                 env_state = self.environment_grabber()
        #             except AttributeError:
        #                 env_state = variables
        #             final_params = []
        #             for para in argg:
        #                 if(type(para) == str):
        #                     variable_value = env_state[para]
        #                     if(isinstance(variable_value,GeneratorType)):
        #                         final_params.append(next(variable_value))
        #                     else:
        #                         final_params.append(variable_value)
        #                 else:
        #                     final_params.append(para)
        #             mean, stdev = final_params                    

        #             yield np.random.normal(loc=mean,scale=stdev)
        #     else:
        #         mean, stdev = argg
        #         while True:
        #             yield np.random.normal(loc=mean,scale=stdev)

        # if(any(type(param) == str for param in argg)):
        #     return normal_generator
        # else:
        #     return normal_generator()

    def logistic(self,argg):
        def logistic_generator(variables = None):
            if(variables):
                while True:
                    try:
                        env_state = self.environment_grabber()
                    except AttributeError:
                        env_state = variables
                    final_params = []
                    for para in argg:
                        if(type(para) == str):
                            variable_value = env_state[para]
                            if(isinstance(variable_value,GeneratorType)):
                                final_params.append(next(variable_value))
                            else:
                                final_params.append(variable_value)
                        else:
                            final_params.append(para)
                    mean, stdev = final_params                    

                    yield np.random.logistic(loc=mean,scale=stdev)
            else:
                mean, stdev = argg
                while True:
                    yield np.random.logistic(loc=mean,scale=stdev)
        if(any(type(param) == str for param in argg)):
            return logistic_generator
        else:
            return logistic_generator()
    def inactive(self, args):
        def empty_generator():
            while True:
                yield None
        return empty_generator()
    def truncated_normal(self, args):
        args = ["truncated_normal"] + args
        genn = self.generator_factory(args)

        return genn
        # def truncnorm_generator(variables = None):
        #     if(variables):
        #         print("2oooo")

        #         while True:
        #             try:
        #                 env_state = self.environment_grabber()
        #             except AttributeError:
        #                 env_state = variables
        #             final_params = []
        #             print(args)
        #             #input("dem args")
        #             for para in args:
        #                 if(type(para) == str):
        #                     variable_value = env_state[para]
        #                     if(isinstance(variable_value,GeneratorType)):
        #                         final_params.append(next(variable_value))
        #                     else:
        #                         final_params.append(variable_value)
        #                 elif(isinstance(para, GeneratorType)):
        #                     final_params.append(next(para))
        #                 elif(callable(para)):
        #                     input("this should only happen once")
        #                     gen_from_callable = para(variables)
        #                     final_params.append(next(gen_from_callable))
        #                     args[args.index(para)] = gen_from_callable #should now remain a generator for next time

        #                 else:
        #                     final_params.append(para)
        #             print(final_params)
        #             input("final_params")
        #             lower, upper, mean, stdev = final_params
                
        #             a, b = (lower - mean) / stdev, (upper - mean) / stdev
                
        #             yield truncnorm.rvs(a,b, loc= mean, scale=stdev)
        #     else:
        #         print("1oooo")
        #         input(args)
        #         final_params = []
        #         for para in args:
        #             if(isinstance(para, GeneratorType)):
        #                 final_params.append(next(para))
        #             else:
        #                 final_params.append(para)
        #         lower, upper, mean, stdev = final_params #this can be simplified by using **kwargs and the correct names from the beginning and then unpacking it into the needed function
        #         a, b = (lower - mean) / stdev, (upper - mean) / stdev
        #         while True:
        #             yield truncnorm.rvs(a,b, loc= mean, scale=stdev)

        # print("looking at the args here")
        # input(args)
        # if(any(type(param) == str for param in args) or any(callable(param) for param in args)):
        #     return truncnorm_generator
        # else:
        #     return truncnorm_generator()
    def constant(self,args):

        def const_generator(variables = None):
            if(variables):
                while True: 
                    #here you need to grab the environment state and incorporate it into the potential yield.
                    #the grabber can be a function...
                    try:
                        env_state = self.environment_grabber()
                    except AttributeError:
                        env_state = variables

                    environment_var = env_state[args[0]]
                    if(isinstance(environment_var,GeneratorType)):
                        yield next(environment_var)
                    else:
                        yield env_state[args[0]]
            else:
                while True: 
                    #print(self.environment_grabber())
                    yield args[0]
        if(type(args[0]) == str):  return const_generator
        else: return const_generator()
        #return const_generator()
    def value(self,args):
        #print(args)
        #input("pause for cause")
        if(len(args) == 1):
            args = args[0]
            try:
                return float(args)
            except (ValueError, TypeError):
                return args
    def userfunction(self,args):
        genn = self.generator_factory(args)

        return genn

        # funcname, *funpara = args

        # try:
        #     my_function = getattr(userfunctions, funcname)
        # except AttributeError:
        #     raise RuntimeError("Used a user-defined function which doesn't exist")
        # def functional_generator(variables = None):
        #     if(variables):
        #         while True:
        #             try:
        #                 env_state = self.environment_grabber()
        #             except AttributeError:
        #                 env_state = variables
        #             final_params = []
        #             for para in funpara:
        #                 if(type(para) == str):
        #                     variable_value = env_state[para]
        #                     if(isinstance(variable_value,GeneratorType)):
        #                         final_params.append(next(variable_value))
        #                     else:
        #                         final_params.append(variable_value)
        #                 else:
        #                     final_params.append(para)
                                   
        #             #input("paaa" + str(one_for_now))
                    
        #             yield my_function(*final_params)
        #     else:
                
        #         #input("paaa2" + str(one_for_now))

        #         while True:
        #             yield my_function(*funpara)
        # if(any(type(param) == str for param in funpara)):
        #     return functional_generator
        # else:
        #     return functional_generator()
            
    

    VARIABLE = str
    indefinite = tuple
    contextpair = tuple    
    CNAME = str
    INT = int
    NUMBER = float
    #contextname = lambda x: str
    lower = value
    upper = value
    stdev = value
    LENGTH = float
    # VARIANCE = float
    mean = value




