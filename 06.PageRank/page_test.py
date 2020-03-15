import networkx as nx
edges = [('A','B'),('A','D'),('A','C'),('B','A'),('B','D'),('C','A'),('D','B'),('D','C')]
G1 = nx.DiGraph()
G1.add_edges_from(edges)
pagerank = nx.pagerank(G1, alpha=0.85)
#for key,value in pagerank.items():
#    print("%s\t%.4f" % (key, value))
print(pagerank)
