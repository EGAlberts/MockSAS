from ast import arg
from scipy.stats import truncnorm
import numpy as np

def myfunction(*args):
    posarg, posarg2 = args
    #print(kwargs)
    #rq_rate = kwargs["request_rate"]
    print(posarg)

    return (posarg + posarg2) /10

def truncated_normal(*args):
    lower, upper, mean, stdev = args

    a, b = (lower - mean) / stdev, (upper - mean) / stdev

    return truncnorm.rvs(a,b, loc= mean, scale=stdev)

def normal(*args):
    mean, stdev = args
    return np.random.normal(loc=mean,scale=stdev)

def uniform(*args):
    lower, upper = args
    return np.random.uniform(low=lower,high=upper)

def truncate(utility):
    bounds = (140,300)

    lower_bound, upper_bound = bounds

    if(utility > upper_bound): utility = upper_bound
        
    elif(utility < lower_bound): utility = lower_bound

    range = upper_bound - lower_bound

    result = float((utility - lower_bound)/range)

    return result


def utilitySWIM(arrival_rate, dimmer, avg_response_time, max_servers, servers):
    OPT_REVENUE = 1.5
    BASIC_REVENUE = 1
    SERVER_COST = 10
    RT_THRESH = 0.75

    ur = arrival_rate * ((1 - dimmer) * BASIC_REVENUE + dimmer * OPT_REVENUE)
    uc = SERVER_COST * (max_servers - servers)
    urt = 1 - ((avg_response_time-RT_THRESH)/RT_THRESH)
	
	
    UPPER_RT_THRESHOLD = RT_THRESH * 4

    delta_threshold = UPPER_RT_THRESHOLD-RT_THRESH

    UrtPosFct = (delta_threshold/RT_THRESH) 

    urt = None
    if(avg_response_time <= UPPER_RT_THRESHOLD):
        urt = ((RT_THRESH - avg_response_time)/RT_THRESH) 
    else: 
        urt = ((RT_THRESH - UPPER_RT_THRESHOLD)/RT_THRESH)

    urt_final = None
    if(avg_response_time <= RT_THRESH):
        urt_final = urt*UrtPosFct 
    else:
        urt_final = urt

    revenue_weight = 0.7
    server_weight = 0.3
    utility = urt_final*((revenue_weight*ur)+(server_weight*uc))

    truncated_reward = truncate(utility)
    return truncated_reward