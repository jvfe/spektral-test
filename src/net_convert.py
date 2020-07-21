# %%
import networkx as nx

# %%
g = nx.read_edgelist("../data/dnabp_rins/just_edges/1A5J.pdb_edges.txt")
scpy = nx.to_scipy_sparse_matrix(g)
