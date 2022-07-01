
from masced_bandits.bandits import init_bandit
from masced_bandits.bandit_options import initialize_arguments
from environmentgrammar import environment_grammar, EnvironmentTransformer
from lark import Lark
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pprint 
from math import floor
from managedsystem import MockSAS
import os
import userfunctions
LINE_DATA_POINTS = 10
TOTAL_ROUNDS = 0
NUM_SEEDS = 100
plt.style.use('seaborn-bright')
parser = Lark(environment_grammar)
mission_statement = "profiles/TestSys.txt"

sys_name = mission_statement.split(".txt")[0]


result_path = "results/" + sys_name + "/" 
os.makedirs(result_path, exist_ok=True)

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

def describe_config(config):
    return str(str(config[0]['name']) + "-" +  str(config[0].get('formula',"")) + str(config[0].get('epsilon',"")) + str(config[0].get('exploration_rounds',"")) + str(config[0].get('decay_rate',"") + str(config[0].get('learning_rate',"")) + str(config[0].get('horizon',""))))


# configs = [[{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "egreedy", 'epsilon': "0.2"}],[{'name': "egreedy", 'epsilon': "0.3"}],[{'name': "egreedy", 'epsilon': "0.4"}],[{'name': "egreedy", 'epsilon': "0.5"}], [{'name': "egreedy", 'epsilon': "0.6"}],[{'name': "egreedy", 'epsilon': "0.7"}],[{'name': "egreedy", 'epsilon': "0.8"}],[{'name': "egreedy", 'epsilon': "0.9"}],[{'name': "egreedy", 'epsilon': "1.0"}], [{'name': "EXP3", 'learning_rate': "0.1"}], [{'name': "EXP3", 'learning_rate': "0.2"}],[{'name': "EXP3", 'learning_rate': "0.3"}], \
#      [{'name': "EXP3", 'learning_rate': "0.4"}], [{'name': "EXP3", 'learning_rate': "0.5"}], [{'name': "EXP3", 'learning_rate': "0.6"}], [{'name': "EXP3", 'learning_rate': "0.7"}], [{'name': "EXP3", 'learning_rate': "0.8"}], [{'name': "EXP3", 'learning_rate': "0.9"}], [{'name': "EXP3", 'horizon': "500"}], 
#       [{'name': "EXP3", 'learning_rate': "0.01"}], [{'name': "EXP3", 'learning_rate': "0.02"}],[{'name': "EXP3", 'learning_rate': "0.03"}], \
#      [{'name': "EXP3", 'learning_rate': "0.04"}], [{'name': "EXP3", 'learning_rate': "0.05"}], [{'name': "EXP3", 'learning_rate': "0.06"}], [{'name': "EXP3", 'learning_rate': "0.07"}], [{'name': "EXP3", 'learning_rate': "0.08"}], [{'name': "EXP3", 'learning_rate': "0.09"}], [{'name': "ETC", 'exploration_rounds': "1"}]]

simple_configs = [ [{'name': "ETC", 'exploration_rounds': "1"}], [{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "egreedy", 'epsilon': "0.2"}],[{'name': "egreedy", 'epsilon': "0.3"}]]

configs = [[{'name': "ETC", 'exploration_rounds': "1"}], [{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "EXP3", 'learning_rate': "0.03"}], [{'name': "EXP3", 'learning_rate': "0.05"}], [{'name': "EXP3", 'learning_rate': "0.07"}], [{'name': "EXP3", 'learning_rate': "0.08"}]]

for config in configs:
    final_res[describe_config(config)] = {"boxplot": [], "lineplot" : []}

for i in range(NUM_SEEDS):
    all_res = {}
    for config in configs:
        #input(str(config) + ">")
        np.random.seed(i * 1339)
        mksas = MockSAS(1,config, parse_tree)
        res = mksas.operation({})
        for key in res.keys():
            #input(str(key) + ">>")
            #input(str(res[key][-1]) + ">>>")
            config_key = describe_config(config)
            final_res[config_key]['boxplot'].append((res[key][-1]))
            every_nth = floor(len(res[key]) / LINE_DATA_POINTS)
            TOTAL_ROUNDS = len(res[key])
            line_points = res[key][::every_nth]
            final_res[config_key]['lineplot'].append(line_points)

#pprint.pprint(final_res)
#print(str(final_res.keys()))
boxplot_data = []
boxplot_labels = []

lineplot_data = {}
#lineplot_labels = []

for config_key in final_res.keys():
    boxplot_data.append(final_res[config_key]["boxplot"])
    boxplot_labels.append(str(config_key))

    stacked_data = final_res[config_key]["lineplot"]
    # pprint.pprint(stacked_data)
    # input("stacked data")
    # for run_tuple in zip(*stacked_data):
    #     pprint.pprint(run_tuple)
    #     input("zip_tuple")
    lineplot_data[str(config_key)] = [mean(run_tuple) for run_tuple in zip(*stacked_data)]


plt.boxplot(boxplot_data, showmeans=True, meanline=True)
plt.ylabel("average reward after " + str(TOTAL_ROUNDS-1) + " rounds (" + str(NUM_SEEDS) + " seeds)", fontdict={'size':14})
plt.xlabel("policies", fontdict={'size':14})

plt.xticks(ticks=list(range(1,len(configs)+1,1)),labels=boxplot_labels)
plt.tight_layout()

plt.savefig(result_path + "boxplot.pdf") 
plt.cla()
for line_key in lineplot_data.keys():
    plt.plot(lineplot_data[line_key], label=str(line_key))

plt.xticks(ticks= list(range(0,LINE_DATA_POINTS+1,1)), labels = [num for num in range(0,TOTAL_ROUNDS+1,floor(TOTAL_ROUNDS / LINE_DATA_POINTS))])
plt.legend(fontsize='medium', title='policy')
plt.ylabel("average reward", fontdict={'size':14})
plt.xlabel("rounds elapsed", fontdict={'size':14})
plt.tight_layout()

plt.savefig(result_path + "lineplot.pdf") 
