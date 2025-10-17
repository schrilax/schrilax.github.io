def addEdge(graph, u, v):
    if u not in graph:
        graph[u] = []
    
    graph[u].append(v)

def topologicalSortUtil(graph, node, visited, stack):
    # Mark the current node as visited.
    visited[node] = True

    if node in graph:
        for i in graph[node]:
            if visited[i] == False:
                topologicalSortUtil(graph, i, visited, stack)
 
    stack.append(node)

def topologicalSort(graph, V):
    visited = [False]*V
    stack = []
 
    for i in range(V):
        if visited[i] == False:
            topologicalSortUtil(graph, i, visited, stack)
 
    return stack[::-1]  # return list in reverse order

graph = {}
V = 6

addEdge(graph, 5, 2)
addEdge(graph, 5, 0)
addEdge(graph, 4, 0)
addEdge(graph, 4, 1)
addEdge(graph, 2, 3)
addEdge(graph, 3, 1)

sorted_vertices = topologicalSort(graph, V)
print(sorted_vertices)