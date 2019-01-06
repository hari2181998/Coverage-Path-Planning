# Python implementation of Kosaraju's algorithm to print all SCCs 

from collections import defaultdict 

#This class represents a directed graph using adjacency list representation 
class Graph: 

	def __init__(self,vertices): 
		self.V= vertices #No. of vertices 
		self.graph = defaultdict(list) # default dictionary to store graph 

	# function to add an edge to graph 
	def addEdge(self,u,v): 
		self.graph[u].append(v) 

	# A function used by DFS 
	def DFSUtil(self,v,visited,a1): 
		# Mark the current node as visited and print it 
		visited[v]= True
		a1.append(v)
		print v
		#Recur for all the vertices adjacent to this vertex 
		for i in self.graph[v]: 
			if visited[i]==False: 
				self.DFSUtil(i,visited,a1)
		return a1 


	def fillOrder(self,v,visited, stack): 
		# Mark the current node as visited 
		visited[v]= True
		#Recur for all the vertices adjacent to this vertex 
		for i in self.graph[v]: 
			if visited[i]==False: 
				self.fillOrder(i, visited, stack) 
		stack = stack.append(v) 
	

	# Function that returns reverse (or transpose) of this graph 
	def getTranspose(self): 
		g = Graph(self.V) 

		# Recur for all the vertices adjacent to this vertex 
		for i in self.graph: 
			for j in self.graph[i]: 
				g.addEdge(j,i) 
		return g 



	# The main function that finds and prints all strongly 
	# connected components 
	def printSCCs(self): 
		
		stack = [] 
		# Mark all the vertices as not visited (For first DFS) 
		visited =[False]*(self.V) 
		# Fill vertices in stack according to their finishing 
		# times 
		for i in range(self.V): 
			if visited[i]==False: 
				self.fillOrder(i, visited, stack) 

		# Create a reversed graph 
		gr = self.getTranspose() 
		
		# Mark all the vertices as not visited (For second DFS) 
		visited =[False]*(self.V) 
		a2=[]
		# Now process all vertices in order defined by Stack 
		while stack: 
			i = stack.pop() 
			if visited[i]==False:
				a1=[] 
				a1=gr.DFSUtil(i, visited,a1)
				a2.append(a1)
				a1=[] 
				print"s"
		return a2 

# Create a graph given in the above diagram 
g = Graph(33)
b=[1, 10, 3, 32, 9, 13, 14, 4]
active_arcs=[(1, 10), (3, 27), (5, 29), (8, 13), (9, 8), (10, 11), (11, 12), (12, 17), (13, 18), (14, 9), (15, 14), (16, 15), (17, 16), (18, 19), (19, 20), (20, 21), (21, 22), (22, 3), (23, 5), (24, 23), (25, 30), (26, 25), (27, 26), (28, 24), (29, 28), (30, 31), (31, 32), (32, 3)]

b=[]
for i in active_arcs:
	if i[0] not in b:
		b.append(i[0])
	if i[1] not in b:
		b.append(i[1])
print(b)
u=b[:]

g= Graph(len(b))
for i in active_arcs:
	z=i[0]
	y=i[1]
	g.addEdge(b.index(z),b.index(y))
a2=g.printSCCs()
for i in range(len(a2)):
	for j in range(len(a2[i])):
		a2[i][j]=b[a2[i][j]]

""" 
g.addEdge(0, 3) 
g.addEdge(3, 2) 
g.addEdge(2, 1) 
g.addEdge(1, 0)
g.addEdge(0,4)
g.addEdge(4,7)
g.addEdge(7,6)
g.addEdge(6,5)
g.addEdge(5,8)
g.addEdge(8,7) 
 """
"""
for i in a:
	x=i[0]
	y=i[1]
	if x==2:
		x=0
	if x==6:
		x=5
	g.addEdge(b.index(x),b.index(y))
	
"""
print ("Following are strongly connected components " +
						"in given graph") 
#a2=g.printSCCs()
"""
for i in range(len(a2)):
	for j in range(len(a2[i])):
		a2[i][j]=b[a2[i][j]]
"""
print(a2) 
#This code is contributed by Neelam Yadav 
