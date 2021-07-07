# python main.py --k 2 --imageloc imge --dataset Advertising.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--imageloc", type=str, required=True)
ap.add_argument("--dataset", type=str, required=True)
args = vars(ap.parse_args())

k = args["k"]
location = args["imageloc"]
dataset = args["dataset"]


class cluster_node:
    def __init__(self, vec, id, left=None, right=None, distance=0.0, node_vector=None):
        self.leftnode = left
        self.rightnode = right
        self.vec = vec
        self.id = id
        self.distance = distance
        if node_vector is None:
            self.node_vector = [self.id]
        else:
            self.node_vector = node_vector[:]


def euclidean_distance(vec1, vec2):
    return np.sqrt(sum((vec1 - vec2) ** 2))


def agglomerative_clustering(data):
    distances = {}
    currentclustid = -1
    nodes = [cluster_node(np.array(data[i]), id=i) for i in range(len(data))]

    while len(nodes) > k:
        lowestpair = (0, 1)
        closest = euclidean_distance(nodes[0].vec, nodes[1].vec)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if (nodes[i].id, nodes[j].id) not in distances:
                    distances[(nodes[i].id, nodes[j].id)] = euclidean_distance(nodes[i].vec, nodes[j].vec)
                d = distances[(nodes[i].id, nodes[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i, j)
        len0 = len(nodes[lowestpair[0]].node_vector)
        len1 = len(nodes[lowestpair[1]].node_vector)
        mean_vector = [(len0 * nodes[lowestpair[0]].vec[i] + len1 * nodes[lowestpair[1]].vec[i]) / (len0 + len1) \
                       for i in range(len(nodes[0].vec))]

        new_node = cluster_node(np.array(mean_vector), currentclustid, left=nodes[lowestpair[0]],
                                right=nodes[lowestpair[1]], \
                                distance=closest,
                                node_vector=nodes[lowestpair[0]].node_vector + nodes[lowestpair[1]].node_vector)
        currentclustid -= 1
        del nodes[lowestpair[1]]
        del nodes[lowestpair[0]]
        nodes.append(new_node)
    return nodes


def main():
    df = pd.read_csv(dataset)
    df = pd.DataFrame(df)
    data = np.array(df)

    f = plt.figure()
    plt.scatter(df[df.columns[0]], df[df.columns[1]])
    orig = location + "/orig.png"
    f.savefig(orig)

    cluster = agglomerative_clustering(data)

    colorset = ['black', 'gray', 'red', 'green', 'blue', 'yellow', 'brown', 'orange']
    j = 0
    m = plt.figure()
    for i in cluster:
        plt.scatter(data[i.node_vector].T[0], data[i.node_vector].T[1], color=colorset[j])
        j += 1
    last = location + "/last.png"
    m.savefig(last)


main()
