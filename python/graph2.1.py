import networkx as nx
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiagent import Country
import copy
from operator import add
import json
import seaborn as sns

def get_pri(g, lang):
    total = 0
    for i in g.nodes:
        total += g.nodes[i]["c"].get_spec_pri_lang(lang)
    return total


def get_sec(g, lang):
    total = 0
    for i in g.nodes:
        total += g.nodes[i]["c"].get_spec_sec_lang(lang)
    return total


def get_all(g, lang):
    return get_sec(g, lang) + get_pri(g, lang)


def get_pri_totals(g, langs):
    totals = []
    for lang in langs:
        totals.append(get_pri(g, lang))
    return totals


def get_sec_totals(g, langs):
    totals = []
    for lang in langs:
        totals.append(get_sec(g, lang))
    return totals


def get_total_totals(g, langs):
    totals = []
    for lang in langs:
        totals.append(get_all(g, lang))
    return totals


def sum_pop(g):
    total = 0
    for i in g.nodes:
        total += g.nodes[i]["c"].get_pop()
    return total


def update_util(g, world_pop):
    for i in g.nodes:
        g.nodes[i]["c"].update_util(world_pop)


def update_weights(g):
    sum_util = 0
    for i in g.nodes:
        sum_util += g.nodes[i]["c"].util

    for i in g.nodes:
        for j in g.nodes:
            if i != j:
                g[i][j]["weight"] = g.nodes[i]["c"].cont * g.nodes[j]["c"].util/sum_util


def update_lang2(g):
    for i in g.nodes:
        g.nodes[i]["c"].update_lang2()


def step(g, half_nbhd, rho):
    # first move people
    for i in g.nodes:
        for j in g.nodes:
            if i != j:
                g.nodes[i]["c"].emmigrate(g.nodes[j]["c"], g[i][j]["weight"])

    # then educate people, calculate births and deaths, and update second language
    for i in g.nodes:
        g.nodes[i]["c"].educate(half_nbhd, rho)
        g.nodes[i]["c"].do_birth()
        g.nodes[i]["c"].do_death()

    # then update each country's util, lang2 and the weights of all edges
    world_p = sum_pop(g)
    update_util(g, world_p)
    update_weights(g)

    # finally unmove all agents
    for i in g.nodes:
        g.nodes[i]["c"].unmove()


def world_census(g):
    for i in g.nodes:
        g.nodes[i]["c"].census()


def iterate(time, g, r, rho, bbc, iter_):
    t = 0
    while t <= time:
        print("        Time elapsed: {0}".format(t+1))

        # update BBC
        for i in g.nodes:
            for lang in bbc[i]["L1"]:
                pop = g.nodes[i]["c"].get_spec_pri_lang(lang)
                bbc[i]["L1"][lang]["AVG"][t] += (pop - bbc[i]["L1"][lang]["AVG"][t]) / iter_
                bbc[i]["L1"][lang]["AVG2"][t] += (pop**2 - bbc[i]["L1"][lang]["AVG2"][t]) / iter_
            for lang in bbc[i]["L2"]:
                pop = g.nodes[i]["c"].get_spec_sec_lang(lang)
                bbc[i]["L2"][lang]["AVG"][t] += (pop - bbc[i]["L2"][lang]["AVG"][t]) / iter_
                bbc[i]["L2"][lang]["AVG2"][t] += (pop**2 - bbc[i]["L2"][lang]["AVG2"][t]) / iter_
            for lang in bbc[i]["T"]:
                pop = g.nodes[i]["c"].get_spec_pri_lang(lang) + g.nodes[i]["c"].get_spec_sec_lang(lang)
                bbc[i]["T"][lang]["AVG"][t] += (pop - bbc[i]["T"][lang]["AVG"][t]) / iter_
                bbc[i]["T"][lang]["AVG2"][t] += (pop**2 - bbc[i]["T"][lang]["AVG2"][t]) / iter_
        step(g, r, rho)
        t += 1
    #world_census(g)

pal = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
       '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
       '#008080', '#e6beff', '#aa6e28', '#fffac8', '#808000',
       '#800000', '#000080']

sns.set_palette(sns.color_palette(pal))

print("Reading Data...")

start = time.time()

df = pd.read_csv("country_data.csv")

cnames = df["Country"].tolist()

cpop_m = df["Population as of 7/1/17 (Millions)"].tolist()

clangs = df["L1"].tolist()

clangs2 = df["L2"].tolist()

cp2 = df["P2"].tolist()

ccont = df["Happiness Index"].tolist()

cbirth = df["Birth Rate (Per Person)"].tolist()

cdeath = df["Death Rate (Per Person)"].tolist()

print("Initializing Network...")

n = len(cnames)

world_pop = sum(cpop_m)

countries = [tuple([cnames[i], {"c": Country(name=cnames[i], lang=clangs[i], p_i=cpop_m[i], world_pop=world_pop,
                                             cont=1/(10*ccont[i]), birth=cbirth[i],
                                             death=cdeath[i], lang2=clangs2[i], p_lang2=cp2[i])}
                    ]) for i in range(n)]

init_world = nx.DiGraph()
init_world.add_nodes_from(countries)
edges = [tuple([x, y]) for x in init_world.nodes for y in init_world.nodes if x != y]
init_world.add_edges_from(edges)

update_weights(init_world)

I = 250             # # ITERATIONS TO RUN
R = 3               # R-NEIGHBORHOOD
RHO = .5/(2 * R)    # PROB TO ADD A LANG ACCUMULATED PER NEIGHBOR *NUMERATOR = MAX OVER NEIGHBORHOOD*
T = 50              # YEARS TO SIMULATE EVERY ITERATION

langs = []
for lang in clangs:
    if lang != "None":
        if lang not in langs:
            langs.append(lang)


langs2 = []
for lang in clangs:
    if lang != "None":
        if lang not in langs2:
            langs2.append(lang)

# The BBC stores average country data over time for all countries and languages
BIG_BOY_COUNTRIES = {"{}".format(i): {"L1": {"{}".format(l1): {"AVG": [0 for _ in range(T+1)],
                                                               "AVG2": [0 for _ in range(T+1)],
                                                               "VAR": [0 for _ in range(T+1)]} for l1 in langs},
                                      "L2": {"{}".format(l2): {"AVG": [0 for _ in range(T+1)],
                                                               "AVG2": [0 for _ in range(T+1)],
                                                               "VAR": [0 for _ in range(T+1)]} for l2 in langs2},
                                      "T": {"{}".format(l): {"AVG": [0 for _ in range(T+1)],
                                                             "AVG2": [0 for _ in range(T+1)],
                                                             "VAR": [0 for _ in range(T+1)]} for l in langs}
                                      } for i in cnames
                     }

BIG_BOY_COUNTRIES["World"] = {"L1": {"{}".format(l1): {"AVG": [0 for _ in range(T+1)],
                                                       "AVG2": [0 for _ in range(T+1)],
                                                       "VAR": [0 for _ in range(T+1)]} for l1 in langs},
                              "L2": {"{}".format(l2): {"AVG": [0 for _ in range(T+1)],
                                                       "AVG2": [0 for _ in range(T+1)],
                                                       "VAR": [0 for _ in range(T+1)]} for l2 in langs2},
                              "T": {"{}".format(l): {"AVG": [0 for _ in range(T+1)],
                                                     "AVG2": [0 for _ in range(T+1)],
                                                     "VAR": [0 for _ in range(T+1)]} for l in langs}
                              }

# These ones are for the bar graphs
avg1 = [0 for i in langs]
avg2 = [0 for j in langs2]
avgt = [0 for k in langs]

i_ = 0
fin = 0

# This is where the fun begins
print("Iterating...")
i_start = time.time()

while i_ < I:
    print("Iteration: {}".format(i_+1))
    world = copy.deepcopy(init_world)

    iterate(T, world, R, RHO, bbc=BIG_BOY_COUNTRIES, iter_=i_+1)

    tot1 = get_pri_totals(world, langs)
    tot2 = get_sec_totals(world, langs2)
    tott = get_total_totals(world, langs)

    avg1 = map(add, tot1, avg1)
    avg2 = map(add, tot2, avg2)
    avgt = map(add, tott, avgt)
    i_ += 1


runtime = time.time() - i_start
print("Completed {0} iterations in {1} seconds".format(I, runtime))

print("Generating bar graphs...")

avg1 = [i/I for i in avg1]
avg2 = [i/I for i in avg2]
avgt = [i/I for i in avgt]

tup1 = [tuple([langs[i], avg1[i]]) for i in range(len(langs))]
tup2 = [tuple([langs2[i], avg2[i]]) for i in range(len(langs2))]
tupt = [tuple([langs[i], avgt[i]]) for i in range(len(langs))]

tup1.sort(key=lambda tup: tup[1], reverse=True)
tup2.sort(key=lambda tup: tup[1], reverse=True)
tupt.sort(key=lambda tup: tup[1], reverse=True)

languages1 = [tup1[i][0] for i in range(len(tup1))]
averages1 = [tup1[i][1] for i in range(len(tup1))]

languages2 = [tup2[i][0] for i in range(len(tup2))]
averages2 = [tup2[i][1] for i in range(len(tup2))]

languagest = [tupt[i][0] for i in range(len(tupt))]
averagest = [tupt[i][1] for i in range(len(tupt))]

x_pos1 = np.arange(len(languages1))
x_pos2 = np.arange(len(languages2))
x_post = np.arange(len(languagest))


plt.bar(x_pos1, averages1, align='center', color='#598381')
plt.xticks(x_pos1, languages1, rotation=50, fontsize=6)
plt.ylabel('Average Number of Speakers (Millions)')
plt.title('Predicted Speakers of Most Popular Native Languages 2067')
plt.savefig("figs/bar_pri.png")

plt.clf()

plt.bar(x_pos2, averages2, align='center', color='#DD9787')
plt.xticks(x_pos2, languages2, rotation=50, fontsize=6)
plt.ylabel('Average Number of Secondary Speakers (Millions)')
plt.title('Predicted Speakers of Most Popular Secondary Languages 2067')
plt.savefig('figs/bar_sec.png')

plt.clf()

plt.bar(x_post, averagest, align='center', color='#166088')
plt.xticks(x_post, languagest, rotation=50, fontsize=6)
plt.ylabel('Average Number of Speakers (Millions)')
plt.title('Predicted Total Speakers of Most Popular Languages 2067')
plt.savefig("figs/bar_tot.png")

plt.clf()

# compute varience of each language in each cat in each country
for c in BIG_BOY_COUNTRIES:
    if c != "World":
        for L in BIG_BOY_COUNTRIES[c]:
            for l in BIG_BOY_COUNTRIES[c][L]:
                for i, v in enumerate(BIG_BOY_COUNTRIES[c][L][l]["VAR"]):
                    v = BIG_BOY_COUNTRIES[c][L][l]["AVG2"][i]-BIG_BOY_COUNTRIES[c][L][l]["AVG"][i]**2

# Since E{X} is linear:
for L in BIG_BOY_COUNTRIES["World"]:
    for l in BIG_BOY_COUNTRIES["World"][L]:
        w_l = [0 for _ in range(T+1)]
        for c in BIG_BOY_COUNTRIES:
            if c != "World":
                w_l = [w_l[i] + BIG_BOY_COUNTRIES[c][L][l]["AVG"][i] for i in range(T+1)]
        BIG_BOY_COUNTRIES["World"][L][l]["AVG"] = w_l
# Unfortunately V{X} is not, so we forgo computing world variances for the sake of computation time
# Generate line graphs
print("Generating line graphs...")
f = 5
for c in BIG_BOY_COUNTRIES:

    for l in BIG_BOY_COUNTRIES[c]["L1"]:
        plt.plot(BIG_BOY_COUNTRIES[c]["L1"][l]["AVG"], label=l)
    plt.title("{0}: Predicted Native Speakers of Most Popular Languages".format(c))
    plt.xticks(np.arange(T+1)[3::f], [2017 + y for y in range(T+1)][3::f])
    plt.xlabel("Year")
    plt.ylabel("Number of Speakers (Millions)")
    plt.legend()
    plt.savefig("figs/{0}_L1.png".format(c))
    plt.clf()

    for l in BIG_BOY_COUNTRIES[c]["L2"]:
        plt.plot(BIG_BOY_COUNTRIES[c]["L2"][l]["AVG"], label=l)
    plt.title("{0}: Predicted Secondary Speakers of Most Popular Languages".format(c))
    plt.xticks(np.arange(T+1)[3::f], [2017 + y for y in range(T+1)][3::f])
    plt.xlabel("Year")
    plt.ylabel("Number of Speakers (Millions)")
    plt.legend()
    plt.savefig("figs/{0}_L2.png".format(c))
    plt.clf()

    for l in BIG_BOY_COUNTRIES[c]["T"]:
        plt.plot(BIG_BOY_COUNTRIES[c]["T"][l]["AVG"], label=l)
    plt.title("{0}: Total Predicted Speakers of Most Popular Languages".format(c))
    plt.xticks(np.arange(T+1)[3::f], [2017 + y for y in range(T+1)][3::f])
    plt.xlabel("Year")
    plt.ylabel("Number of Speakers (Millions)")
    plt.legend()
    plt.savefig("figs/{0}_T.png".format(c))
    plt.clf()

# Save BBC
timestamp = time.time()
print("Writing BIG BOY to json for posterity... (Timestamp {})".format(timestamp))
with open('data/BIG_BOY_NEW_{}.json'.format(timestamp), 'w') as fp:
    json.dump(BIG_BOY_COUNTRIES, fp)

runtime = time.time() - start

print("Total program time: {} seconds".format(runtime))
