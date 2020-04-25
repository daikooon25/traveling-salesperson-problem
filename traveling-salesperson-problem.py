# Copyright 2020 Sigma-i Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from copy import deepcopy

from pyqubo import Array, Placeholder, Constraint
from dwave.system import DWaveSampler
from minorminer import find_embedding
import dimod
from dwave.embedding import embed_qubo, unembed_sampleset
from dwave.embedding.chain_breaks import majority_vote, discard, weighted_random, MinimizeEnergy



# --- Problem setting ---

# Define the coordinates of each city and one origin city (at random in this demo)
N = 8

X_range = Y_range = 500
x_pos = [np.random.randint(0, X_range) for _ in range(N)]
y_pos = [np.random.randint(0, Y_range) for _ in range(N)]
positions = {i: (x_pos[i], y_pos[i]) for i in range(N)}  # you can rewrite this line

# Choose origin (and end) city and fix it
origin = np.random.choice(np.arange(N))
origin_pos = positions[origin]

others = list(range(N))
others.remove(origin)

# Set a graph
G = nx.Graph()
G.add_nodes_from(np.arange(N))

# Calculate the distance between each city
distances = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        distances[i][j] = np.sqrt((x_pos[i] - x_pos[j])**2 + (y_pos[i] - y_pos[j])**2)
        distances[j][i] = distances[i][j]


# --- Problem formulation ---

# Use pyqubo package
q = Array.create('q', shape=(N-1, N-1), vartype='BINARY')

def normalize(exp):
    """ Normalization function """
    qubo, offset = exp.compile().to_qubo()
    
    max_coeff = abs(np.max(list(qubo.values())))
    min_coeff = abs(np.min(list(qubo.values())))
    norm = max_coeff if max_coeff - min_coeff > 0 else min_coeff
    
    return exp / norm

# Cost function
exp_origin = sum(distances[origin][others[i]]*1*q[i][0] + 
             distances[others[i]][origin]*q[i][N-2]*1 for i in range(N-1))
exp_others = sum(distances[others[i]][others[j]]*q[i][t]*q[j][t+1]
              for i in range(N-1) for j in range(N-1) for t in range(N-2))
H_cost = normalize(exp_origin + exp_others)

# Constraint
H_city = Constraint(normalize(sum((sum(q[i][t] for t in range(N-1))-1)**2 for i in range(N-1))), 'city')
H_time = Constraint(normalize(sum((sum(q[i][t] for i in range(N-1))-1)**2 for t in range(N-1))), 'time')

# Express objective function and compile it to model
H_obj = H_cost + Placeholder('lam') * (H_city + H_time)
model = H_obj.compile()


# --- Solve QUBO and unembed samplesn ---

# Get the QUBO matrix from the model
feed_dict = {'lam':5.0}  # the coefficient of constraints
qubo, offset = model.to_qubo(feed_dict=feed_dict)

adjacency = {key: 1.0 for key in qubo.keys() if key[0]!=key[1]}

# Run QUBO on DW_2000Q_5
sampler = DWaveSampler(solver='DW_2000Q_5')
embedding = find_embedding(adjacency, sampler.edgelist)
qubo_emb = embed_qubo(qubo, embedding, sampler.adjacency)
response = sampler.sample_qubo(qubo_emb, num_reads=1000, postprocess='optimization')

# Unembed samples by specifying chain_break_method argument of unembed_sampleset
bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    # majority_vote (default)
sample_majority_vote = unembed_sampleset(response, embedding, bqm, chain_break_method=majority_vote)
    # discard
sample_discard = unembed_sampleset(response, embedding, bqm, chain_break_method=discard)
    # weighted_random
sample_weighted_random = unembed_sampleset(response, embedding, bqm, chain_break_method=weighted_random)
    # MinimizeEnergy
cbm = MinimizeEnergy(bqm, embedding)
sample_MinimizeEnergy = unembed_sampleset(response, embedding, bqm, chain_break_method=cbm)

# Check if unembeded solutions with each method satisfy constrains
def check_constraint(samples):
    """Decode solutions and check constraints """
    if samples.record.shape[0] != 0:
        count = 0
        for s in samples.record['sample']:
            sample_dict = {idx: s[i] for i,idx in enumerate(samples.variables)}
            decoded, broken, energy = model.decode_solution(sample_dict, 'BINARY', feed_dict=feed_dict) 
            if broken == {}:
                count += 1
        return count/1000
    else:
        return None

sampleset = [sample_majority_vote, sample_discard, sample_weighted_random, sample_MinimizeEnergy]
names = ['majority_vote', 'discard', 'weighted_random', 'minimize_energy']
satisfy = []
print('the rate of solution satisfying constraints:')
for i,samples in enumerate(sampleset):
    satisfy.append(check_constraint(samples))
    print(' ' + names[i] + ': ', satisfy[i])


# --- Visualize the result ---

# Show the histogram of energy in each method
names = ['majority_vote', 'discard', 'weighted_random', 'MinimizeEnergy']
for i in range(len(sampleset)):
    plt.hist(sampleset[i].record['energy'], alpha=0.7, label=names[i])
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.legend()
plt.title('Energy distribution in each chain_break_method', y=1.05)
plt.savefig('energy_distribution.png')

def create_order(solution_arr):
    """Create an array which shows traveling order from the solution"""
    order = [origin]
    for li in solution_arr.T:
        cities = np.where(li)[0].tolist()
        if cities == []:
            continue
        if len(cities) > 1:
            order.append(others[np.random.choice(cities, 1)[0]])
        else:
            order.append(others[cities[0]])
    return order

def show_result(solution_arr, name):
    """Show a result in two forms"""
    fig = plt.figure(figsize=(10, 5))
    grid = GridSpec(1,5)
    
    # Plot the traveling order
    add_zero = np.append(np.zeros((N-1,1)), solution_arr, axis=1) 
    add_origin = np.insert(add_zero, origin, np.append(1,np.zeros(N-1)), axis=0)
    
    ax1 = fig.add_subplot(grid[0,0:2])
    ax1.imshow(add_origin)
    ax1.set_xlabel('Order')
    ax1.set_ylabel('Cities')
    ax1.set_xticks(np.arange(N))
    ax1.set_yticks(np.arange(N))
    
    # Draw the traveling route on graph G
    order = create_order(solution_arr)
    edges = []
    edges += [(order[t], order[t+1]) for t in range(len(order)-1)]
    edges += [(order[-1], origin)]
    
    G_copy = deepcopy(G)
    G_copy.add_edges_from(edges)
    ax2 = fig.add_subplot(grid[0,2:])
    nx.draw(G_copy, positions, node_color=['red' if i == origin else 'blue' for i in range(N)], with_labels=True, ax=ax2)
    
    fig.tight_layout()
    fig.suptitle(name, fontsize=14)
    fig.subplots_adjust(top=0.9)

# Show the result of samples with lowest energy
for samples, name in zip(sampleset, names):
    lowest = samples.lowest().record['sample']
    if len(lowest) == 0:
        continue
    else:
        solution = lowest[0].reshape(N-1, N-1)
        show_result(solution, name)
        plt.savefig(name + '.png')
