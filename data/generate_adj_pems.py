import torch
import networkx as nx


Adj_file = 'Adj(PeMS).txt'
def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.Graph())
    return G


def read_adj(Adj_file):
    nx_G = read_graph(Adj_file)
    n = nx_G.number_of_nodes()
    adj_weight = torch.zeros(n, n)
    adj_rela = torch.zeros(n, n)
    for (x,y) in nx_G.edges:
        adj_weight[x][y] += nx_G[x][y]['weight'] if nx_G[x][y]['weight'] > 0 else 0
        adj_rela[x][y] += 1 if nx_G[x][y]['weight'] > 0 else 0
    print('return tensor adj:', adj_rela.shape)
    return adj_weight.float(), adj_rela.float()

if __name__ == '__main__':
    read_adj(Adj_file)