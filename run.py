
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
import sys
import csv
LINE_DATA_POINTS = 10
TOTAL_ROUNDS = 0
NUM_SEEDS = 30

SWIM_ORDERING = {"UCB-TN" : 1,
"egreedy-0.2" : 2,
"DUCB-0.997" : 3,
"DUCB-0.995" : 4,
"egreedy-0.4" : 5,
"EXP3-333" : 6,
"DUCB-0.992" : 7,
"DUCB-0.99" : 8,
"egreedy-0.6" :9 ,
"egreedy-0.8" : 10,
"DUCB-0.97" : 11,
"DUCB-0.95" : 12,
"egreedy-1.0" : 13,
"DUCB-0.92" : 14,
"DUCB-0.89" : 15}
plt.style.use('seaborn-bright')
parser = Lark(environment_grammar)
mission_statement = "profiles/SWIMProfile.txt"

sys_name = mission_statement.split(".txt")[0]

result_path = "results/" + sys_name + "/" 
os.makedirs(result_path, exist_ok=True)

NUM_SEEDS = int(sys.argv[1])
CSV_name = str(sys.argv[2])


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
    return str(str(config['name']) + "-" +  str(config.get('formula',"")) + str(config.get('epsilon',"")) + str(config.get('exploration_rounds',"")) + str(config.get('decay_rate',"")) + str(config.get('learning_rate',"")) + str(config.get('horizon',"")) + str(config.get('gamma',"")) )


# egreedies = [[{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "egreedy", 'epsilon': "0.2"}],[{'name': "egreedy", 'epsilon': "0.3"}], \
# [{'name': "egreedy", 'epsilon': "0.4"}],[{'name': "egreedy", 'epsilon': "0.5"}], [{'name': "egreedy", 'epsilon': "0.6"}],[{'name': "egreedy", 'epsilon': "0.7"}], \
# [{'name': "egreedy", 'epsilon': "0.8"}],[{'name': "egreedy", 'epsilon': "0.9"}],[{'name': "egreedy", 'epsilon': "1.0"}]]

egreedies = [{'name': "egreedy", 'epsilon': "0.2"}, {'name': "egreedy", 'epsilon': "0.4"},{'name': "egreedy", 'epsilon': "0.6"}, \
{'name': "egreedy", 'epsilon': "0.8"},{'name': "egreedy", 'epsilon': "1.0"}]

ucbs = [{'name': "UCB", 'formula': "TN"}]


# exp3s = [[{'name': "EXP3", 'learning_rate': "0.1"}], \
# [{'name': "EXP3", 'learning_rate': "0.2"}],[{'name': "EXP3", 'learning_rate': "0.3"}], [{'name': "EXP3", 'learning_rate': "0.4"}], [{'name': "EXP3", 'learning_rate': "0.5"}],\
# [{'name': "EXP3", 'learning_rate': "0.6"}], [{'name': "EXP3", 'learning_rate': "0.7"}], [{'name': "EXP3", 'learning_rate': "0.8"}], [{'name': "EXP3", 'learning_rate': "0.9"}], \
# [{'name': "EXP3", 'learning_rate': "1.0"}]]   

exp3s =  [{'name': "EXP3", 'horizon': "333"}]

discountucbs = [{'name': "DUCB", 'gamma': "0.89"}, {'name': "DUCB", 'gamma': "0.92"}, {'name': "DUCB", 'gamma': "0.95"}, {'name': "DUCB", 'gamma': "0.97"}, {'name': "DUCB", 'gamma': "0.99"},\
    {'name': "DUCB", 'gamma': "0.992"}, {'name': "DUCB", 'gamma': "0.995"}, {'name': "DUCB", 'gamma': "0.997"}]

configs = egreedies + ucbs + exp3s + discountucbs

#simple_configs = [ [{'name': "ETC", 'exploration_rounds': "1"}], [{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "egreedy", 'epsilon': "0.2"}],[{'name': "egreedy", 'epsilon': "0.3"}]]

#configs = [[{'name': "ETC", 'exploration_rounds': "1"}], [{'name': "UCB", 'formula': "TN"}], [{'name': "egreedy", 'epsilon': "0.1"}], [{'name': "EXP3", 'learning_rate': "0.03"}], [{'name': "EXP3", 'learning_rate': "0.05"}], [{'name': "EXP3", 'learning_rate': "0.07"}], [{'name': "EXP3", 'learning_rate': "0.08"}]]

for config in configs:
    final_res[describe_config(config)] = {"boxplot": [], "lineplot" : [], "positions": [], "match_swim": []}

for i in range(NUM_SEEDS):
    all_res = {}
    for config in configs:
        #input(str(config) + ">")
        np.random.seed(i * 293019)
        mksas = MockSAS(config, parse_tree)
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
        
    position_dict = {}
    for config in configs:
        for key in res.keys():
            config_key = describe_config(config)
            position_dict[config_key] = final_res[config_key]['boxplot'][-1]

    ranking = (list({k: v for k, v in sorted(position_dict.items(), key=lambda item: item[1], reverse= True)}.keys()))

    for policy in ranking:
        final_res[policy]["match_swim"].append(1 if ((ranking.index(policy)+1) == SWIM_ORDERING[policy]) else 0)
        final_res[policy]["positions"].append(ranking.index(policy)+1)


#SWIM

#

#pprint.pprint(final_res)
#print(str(final_res.keys()))

boxplot_data = []
boxplot_labels = []

lineplot_data = {}
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

row_titles = ["Policy Name", "Mean", "Median", "Upper Q", "Lower Q", "IQR", "Upper W", "Lower W", "Average Ranking", "Percentage Matching SWIM"]
all_rows = [row_titles]
for i, datum in enumerate(boxplot_data):

    entry_row = []

    entry_row.append(boxplot_labels[i])

    data_mean = np.mean(datum)
    median = np.median(datum)
    upper_quartile = np.percentile(datum, 75)
    lower_quartile = np.percentile(datum, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = np.array(datum)[datum<=upper_quartile+1.5*iqr].max()
    lower_whisker = np.array(datum)[datum>=lower_quartile-1.5*iqr].min()

    average_rank = np.mean(final_res[boxplot_labels[i]]["positions"])
    perct_match = str(np.mean(final_res[boxplot_labels[i]]["match_swim"]) * 100) + "%"


    entry_row+=[median,data_mean,upper_quartile,lower_quartile,iqr,upper_whisker,lower_whisker, average_rank, perct_match]

    all_rows.append(entry_row)




with open(result_path + CSV_name, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    for the_row in all_rows:
        spamwriter.writerow(the_row)




def boxplotter(boxplot_data, boxplot_labels, result_path):
    plt.boxplot(boxplot_data, showmeans=True, meanline=True)
    plt.ylabel("average reward after " + str(TOTAL_ROUNDS-1) + " rounds (" + str(NUM_SEEDS) + " seeds)", fontdict={'size':14})
    plt.xlabel("policies", fontdict={'size':14})

    plt.xticks(ticks=list(range(1,len(configs)+1,1)),labels=boxplot_labels)
    plt.tight_layout()

    plt.savefig(result_path + "boxplot.pdf") 
    plt.cla()

def lineplotter(lineplot_data,result_path):
    for line_key in lineplot_data.keys():
        plt.plot(lineplot_data[line_key], label=str(line_key))

    plt.xticks(ticks= list(range(0,LINE_DATA_POINTS+1,1)), labels = [num for num in range(0,TOTAL_ROUNDS+1,floor(TOTAL_ROUNDS / LINE_DATA_POINTS))])
    plt.legend(fontsize='medium', title='policy')
    plt.ylabel("average reward", fontdict={'size':14})
    plt.xlabel("rounds elapsed", fontdict={'size':14})
    plt.tight_layout()

    plt.savefig(result_path + "lineplot.pdf") 
    plt.cla()
