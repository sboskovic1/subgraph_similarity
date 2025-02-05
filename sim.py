import networkx as nx
from itertools import combinations
from collections import Counter
import numpy as np
import statistics
import random
import time

def parse_labels(file):
    labels = []
    with open(file, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return labels

def parse_A(): 
    adj = []
    max = 0
    with open('MOLT-4/MOLT-4_A.txt', 'r') as f:
        for line in f:
            row, col = map(int, line.strip().split(','))
            adj.append((row, col))
            if row > max:
                max = row
    return adj

def create_graphs(adjacency_matrix, indicators, e_labels, g_labels, n_labels):
    graphs = {}
    for i, (r, c) in enumerate(adjacency_matrix):
        indicator = indicators[r - 1]
        label = g_labels[indicator - 1]

        if label not in graphs:
            graphs[label] = {}
        if indicator not in graphs[label]:
            graphs[label][indicator] = nx.Graph()
            graphs[label][indicator].graph['label'] = label

        graph = graphs[label][indicator]
        graph.add_edge(r, c, label=e_labels[i])

        if 'label' not in graph.nodes[r]:
            graph.nodes[r]['label'] = n_labels[r - 1]
        if 'label' not in graph.nodes[c]:
            graph.nodes[c]['label'] = n_labels[c - 1]
        
    return graphs

def draw_graph():
    pass

def get_data(graphs, k):
    data = {}
    times = {}

    for i in range(k):
        start = time.time()
        data0, data1, data_both = get_similarity(graphs[0], graphs[1], i + 1)
        data[i + 1] = [data0, data1, data_both]
        end = time.time()
        times[i + 1] = end - start
    return data, times

def get_similarity(g0, g1, k):
    data0 = []
    data1 = []
    data_both = []

    combinations_0 = list(g0.items())
    combinations_1 = list(g1.items())
    random.shuffle(combinations_0)
    random.shuffle(combinations_1)
    for i in range(3):
        graphs1 = combinations_0[i][1]
        graphs2 = combinations_0[i + 1][1]
        sim = calc_kernel(graphs1, graphs2, k)
        data0.append(sim)

    for i in range(3):
        graphs1 = combinations_1[i][1]
        graphs2 = combinations_1[i + 1][1]
        sim = calc_kernel(graphs1, graphs2, k)
        data1.append(sim)

    combinations_0 = list(g0.items())
    combinations_1 = list(g1.items())
    random.shuffle(combinations_0)
    random.shuffle(combinations_1)
    for i in range(3):
        graphs1 = combinations_0[i][1]
        graphs2 = combinations_1[i][1]
        sim = calc_kernel(graphs1, graphs2, k)
        data_both.append(sim)

    return data0, data1, data_both

def calc_kernel(graph1, graph2, k):
    count_g1 = get_freq(graph1, k)
    count_g2 = get_freq(graph2, k)

    all_subgraphs = set(count_g1.keys()).union(count_g2.keys())
    vector_g1 = np.array([count_g1.get(subgraph, 0) for subgraph in all_subgraphs])
    vector_g2 = np.array([count_g2.get(subgraph, 0) for subgraph in all_subgraphs])

    return np.dot(vector_g1, vector_g2)

def get_freq(graph, k):
    freq = Counter()
    for nodes in combinations(graph.nodes, k):
        subgraph = graph.subgraph(nodes)
        if nx.is_connected(subgraph):
            node_labels = [graph.nodes[n]['label'] for n in subgraph.nodes()]
            edge_labels = []
            for x, y, z in subgraph.edges(data=True):
                edge_labels.append((graph.nodes[x]['label'], graph.nodes[y]['label'], z['label']))
            labels = (tuple(sorted(node_labels)), tuple(sorted(edge_labels)))
            freq[labels] += 1
    return freq

if __name__ == "__main__":
    adjacency_matrix = parse_A()
    indicators = parse_labels('MOLT-4/MOLT-4_graph_indicator.txt')
    e_labels = parse_labels('MOLT-4/MOLT-4_edge_labels.txt')
    g_labels = parse_labels('MOLT-4/MOLT-4_graph_labels.txt')
    n_labels = parse_labels('MOLT-4/MOLT-4_node_labels.txt')

    graphs = create_graphs(adjacency_matrix, indicators, e_labels, g_labels, n_labels)

    data, times = get_data(graphs, 5)
    for k, (data0, data1, data_both) in data.items():
        print(f'k: {k}')
        print(f'Label 0: mean={statistics.mean(data0)} stdev={statistics.stdev(data0)}')
        print(f'Label 1: mean={statistics.mean(data1)} stdev={statistics.stdev(data1)}')
        print(f'Label 0 vs Label 1: mean={statistics.mean(data_both)} stdev={statistics.stdev(data_both)}')
        print(f'Time: {times[k]}')

# n = 50, k = 4
# k: 1
# Label 0: mean=354 stdev=226.2144999773445
# Label 1: mean=534 stdev=295.1270912674741
# Label 0 vs Label 1: mean=521 stdev=342.83961264708023
# Time: 0.5325441360473633
# k: 2
# Label 0: mean=168 stdev=142.51666569212176
# Label 1: mean=390 stdev=408.43848986108054
# Label 0 vs Label 1: mean=164 stdev=81.77407902263406
# Time: 1.8687725067138672
# k: 3
# Label 0: mean=165 stdev=146.47866738880444
# Label 1: mean=534 stdev=932.6762568008259
# Label 0 vs Label 1: mean=258 stdev=211.6719159454083
# Time: 22.578144550323486
# k: 4
# Label 0: mean=275 stdev=294.7286887970019
# Label 1: mean=538 stdev=511.6404987879673
# Label 0 vs Label 1: mean=328 stdev=404.49845487962006
# Time: 315.40502429008484

# n = 200, k = 3
# k: 1
# Label 0: mean=386 stdev=207.34271147064706
# Label 1: mean=533 stdev=147.20733677368122
# Label 0 vs Label 1: mean=439 stdev=235.45063176810547
# Time: 1.0503709316253662
# k: 2
# Label 0: mean=183 stdev=129.6919426949878
# Label 1: mean=301 stdev=236.4466113100376
# Label 0 vs Label 1: mean=251 stdev=224.49498880821372
# Time: 7.685718774795532
# k: 3
# Label 0: mean=210 stdev=166.9820349618485
# Label 1: mean=354 stdev=289.0795738200816
# Label 0 vs Label 1: mean=245 stdev=182.11260252931427
# Time: 100.72726202011108

# n = 3, k = 5
# k: 1
# Label 0: mean=609 stdev=481.5828070020773
# Label 1: mean=299 stdev=97.98469268207153
# Label 0 vs Label 1: mean=450 stdev=210.57065322594218
# Time: 0.0454409122467041
# k: 2
# Label 0: mean=113 stdev=102.04900783447137
# Label 1: mean=72 stdev=48.28043081829324
# Label 0 vs Label 1: mean=216 stdev=124.53112060846478
# Time: 0.3763561248779297
# k: 3
# Label 0: mean=87 stdev=85.11169132381285
# Label 1: mean=288 stdev=215.32301316858818
# Label 0 vs Label 1: mean=468 stdev=450.3620765561861
# Time: 1.8666281700134277
# k: 4
# Label 0: mean=174 stdev=60.473134530963414
# Label 1: mean=380 stdev=194.65867563507155
# Label 0 vs Label 1: mean=337 stdev=198.44394674567425
# Time: 7.681243658065796
# k: 5
# Label 0: mean=110 stdev=151.24152868838638
# Label 1: mean=585 stdev=260.3497647396671
# Label 0 vs Label 1: mean=48 stdev=67.30527468185537
# Time: 75.67859601974487
            
