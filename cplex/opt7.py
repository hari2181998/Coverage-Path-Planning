from __future__ import print_function

import sys

import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
import numpy as np 
def powerset(A):
	    if A == []:
	        return [[]]
	    a = A[0]
	    incomplete_pset = powerset(A[1:])
	    rest = []
	    for set in incomplete_pset:
	        rest.append([a] + set)
	    a=(rest + incomplete_pset)
	    return a

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
		#print (v ,)
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
				#print"s"
		return a2


class LazyCallback(LazyConstraintCallback):

	

	def __init__(self, env) :
		LazyConstraintCallback.__init__(self, env)

	def __call__(self):
		qq=open("dataopt.txt","w")
		#print("sssss")
		#print(self.get_objective_value())
		active_arcs=[]
		for i in self.D:
			for j in self.D:
				print(self.get_values(self.w[i][j]),i,j)
				#print(self.get_values(x[1][2]),i,j)
						
		for i in self.A:
			print()
			if self.get_values(self.x[i[0]][i[1]])>0.9:
				active_arcs.append(i)
				#print("s")
		print(active_arcs)
		plt.scatter(loc_x[1:],loc_y[1:],c='b')
		
		
		r=[]
		t1=[]
		dpair=[]
		for i in active_arcs:
			if i[0] in self.D:
				r.append(i[0])
			if i[1] in self.D:
				t1.append(i[1])
		for i in r:
			for j in t1:
				if i!=j and [i,j] not in dpair:
					dpair.append([i,j])
		print(dpair)

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
		print(a2)
		if len(a2)!=1:
			for i in a2:
				if len(i)!=1:
					thevars=[]
					thecoefs=[]
					pp=0
					for j in i:
						for k in self.V:
							if k not in i:
								#print(x[k][j])
								thevars.append(self.x[k][j])
								thevars.append(self.x[j][k])
								thecoefs.append(1)
								thecoefs.append(1)
								pp+=self.get_values(self.x[k][j])+self.get_values(self.x[j][k])
					if pp<0.90:
						print(i)
						self.add(constraint=cplex.SparsePair(thevars,thecoefs),
					 			sense="G",
					 			rhs=1)
					for j in i:
						if j not in self.D:
							b.remove(j)



			print(b)
			for w1 in dpair:
				#print(k)
				c=b[:]
				"""for i in active_arcs:
					if i[0] not in c and i[0]!=w[1]:
						c.append(i[0])
					if i[1] not in c and i[1]!=w[1]:
						c.append(i[1])"""
				c.remove(w1[1])
				#d=c[0:]
				#print(c)
				#c[0]=2
				#print(c)
				print(c)
				print(c[0])
				
				g=Graph(len(c))
				for i in active_arcs:
					z=i[0]
					y=i[1]
					
					if y==w1[1]:
						y=w1[0]
					if z==w1[1]:
						z=w1[0]
					if z not in c or y not in c:
						continue
					#if z==c[0]:
					#	g.addEdge(0,c.index(y))
					#else:	
					#print(c.index(z))
					g.addEdge(c.index(z),c.index(y))
				a2=g.printSCCs()
				print(a2,"rrr")
				
				
				for i in range(len(a2)):
					for j in range(len(a2[i])):
						a2[i][j]=c[a2[i][j]]
				

				for i in a2:
					if w1[0] in i and len(i)>1:
						i.append(w1[1])
				print(a2,"ttt")
				if len(a2)==1:
					break
				
				if len(a2)!=1:
					for i in a2:
						
						if len(i)!=1:
							print(i)
							
							for jj in range(0,10):
								for ii in active_arcs:
									#print(ii)
									if ii[0] in i and ii[1] not in i:
										i.append(ii[1])
										print(ii)
									if ii[1] in i and ii[0] not in i:
										i.append(ii[0])
										print(ii)
										
							if len(i)==len(u):
								pp=1
								plt.scatter(loc_x[1:],loc_y[1:],c='b')
								#for i,j in active_arcs:
									#plt.plot([self.loc_x[i],self.loc_x[j]],[self.loc_y[i],self.loc_y[j]],c='g')
								#plt.axis('equal')
								#plt.show()
								break

							pp=0
							thevars=[]
							thecoefs=[]
							for j in i:
								for k in self.V:
									if k not in i:
										#print(x[k][j])
										thevars.append(self.x[k][j])
										thevars.append(self.x[j][k])
										thecoefs.append(1)
										thecoefs.append(1)
										pp+=self.get_values(self.x[k][j])+self.get_values(self.x[j][k])
							if pp<0.9:
								print(i,"yyy")
								self.add(constraint=cplex.SparsePair(thevars,thecoefs),
							 			sense="G",
										rhs=1)
		"""
		if pp>0.9:
			#plt.show()
			qq.write(str(active_arcs))
			qq.write("\n")
			qq.write("    ssss       ")
			qq.close()
			#for i,j in active_arcs:
			#	plt.plot([self.loc_x[i],self.loc_x[j]],[self.loc_y[i],self.loc_y[j]],c='g')
			#plt.axis('equal')
			#plt.show()
		"""

						 			
						 			
						 			
#assign one variable to each depot
#if one of them is 1 all others should be zero condition

		

						







rnd = np.random
rnd.seed(0)
n=81
Q=150
X=20
a=40

b=4.3
nd=8
coords=[[-a/2,a/2],[0,a/2],[a/2,a/2],[a/2,0],[a/2,-a/2],[0,-a/2],[-a/2,-a/2],[-a/2,0]]
loc_x=[-a/2,0,a/2,a/2,a/2,0,-a/2,-a/2]
loc_y=[a/2,a/2,a/2,0,-a/2,-a/2,-a/2,0]
#random.shuffle(self.coords)
for i in range(0,int(a/b)):
  for j in range(0,int(a/b)):

    coords.append([-a/2+b/2+(b*j),a/2-b/2-(b*i)])
    loc_x.append(-a/2+b/2+(b*j))
    loc_y.append(a/2-b/2-(b*i))
print(coords)
print(loc_x)
print(loc_y)
D= range(0,nd)
N=[i for i in range(nd,n+nd)]
V=range(0,nd)+N
print(V)
import matplotlib.pyplot as plt
A=[(i,j) for i in V for j in V ]
#print(A)
c={(i,j):np.hypot(loc_x[i]-loc_x[j],loc_y[i]-loc_y[j]) for i,j in A}
for i in D:
	for j in D:
		c[i,j]=abs(loc_x[i]-loc_x[j])+abs(loc_y[i]-loc_y[j])
cpx = cplex.Cplex()
#print(len(A))
x=[]
#print(c[0][2])
for i in range(len(V)):
    x.append([])
    for j in range(len(V)):
    	
    		#print (i,j)
    		#print(c[i][j])
	        varName = "x." + str(i) + "." + str(j)
	        x[i].append(cpx.variables.get_num())
	        cpx.variables.add(obj=[c[i,j]],
	                          lb=[0.0], ub=[1.0], types=["B"],
	                          names=[varName])
    cpx.variables.set_upper_bounds(x[i][i], 0)

w=[]
for i in range(len(D)):
    w.append([])
    for j in range(len(D)):
    	
    		#print (i,j)
    		#print(c[i][j])
	        varName = "w." + str(i) + "." + str(j)
	        w[i].append(cpx.variables.get_num())
	        cpx.variables.add(obj=[c[i,j]],
	                          lb=[0.0], ub=[8.0], types=["C"],
	                          names=[varName])
print(w)
t=cpx.variables.add(obj=D,
                         lb=[-1] * len(D), ub=[1] * len(D),
                         types=['C'] * len(D),
                         names=['t(%d)' % (j) for j in D]
                         )


                         

p=cpx.variables.add(obj=V,
                         lb=[-X] * len(V), ub=[X] * len(V),
                         types=['C'] * len(V),
                         names=['p(%d)' % (j) for j in V]
                         )
o=cpx.variables.add(obj=V,
                         lb=[-X] * len(V), ub=[X] * len(V),
                         types=['C'] * len(V),
                         names=['o(%d)' % (j) for j in V]
                         )

u = cpx.variables.add(obj=V,
                         lb=[0] * len(V), ub=[Q] * len(V),
                         types=['C'] * len(V),
                         names=['u(%d)' % (j) for j in V]
                         )
a=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['a(%d)' % (j) for j in V]
                    )
b=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['b(%d)' % (j) for j in V]
                    )
d=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['d(%d)' % (j) for j in V]
                    )
e=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['e(%d)' % (j) for j in V]
                    )
f=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['f(%d)' % (j) for j in V]
                    )
g=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['g(%d)' % (j) for j in V]
                    )
h=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['h(%d)' % (j) for j in V]
                    )
ii=cpx.variables.add(obj=V,
                    lb=[0.0] * len(V), ub=[1.0] * len(V),
                    types=['B'] * len(V),
                    names=['ii(%d)' % (j) for j in V]
                    )

for i in V:
	cpx.objective.set_linear(u[i],0)
	cpx.objective.set_linear(p[i],0)
	cpx.objective.set_linear(o[i],0)
	cpx.objective.set_linear(a[i],0)
	cpx.objective.set_linear(b[i],0)
	cpx.objective.set_linear(d[i],0)
	cpx.objective.set_linear(e[i],0)
	cpx.objective.set_linear(f[i],0)
	cpx.objective.set_linear(g[i],0)
	cpx.objective.set_linear(h[i],0)
	cpx.objective.set_linear(ii[i],0)
	
	
for i in D:
	cpx.objective.set_linear(t[i],0)
	cpx.variables.set_upper_bounds(u[i], 0)
	cpx.variables.set_upper_bounds(p[i], 0)
	cpx.variables.set_upper_bounds(o[i], 0)
	cpx.variables.set_upper_bounds(a[i], 0)
	cpx.variables.set_upper_bounds(b[i], 0)
	cpx.variables.set_upper_bounds(d[i], 0)
	cpx.variables.set_upper_bounds(e[i], 0)
	cpx.variables.set_upper_bounds(f[i], 0)
	cpx.variables.set_upper_bounds(g[i], 0)
	cpx.variables.set_upper_bounds(h[i], 0)
	cpx.variables.set_upper_bounds(ii[i], 0)
for i in D:
	for j in D:
		cpx.variables.set_upper_bounds(x[i][j],0)
	
print(e)

for i in N:
    thevars = []
    thecoefs = []
    for j in range(0, len(V)):
    	if i!=j:
        	thevars.append(x[i][j])
        	thecoefs.append(1)
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
        senses=["E"], rhs=[1.0])
for j in N:
    thevars = []
    thecoefs = []
    for i in range(0, len(V)):
    	if i!=j:
        	thevars.append(x[i][j])
        	thecoefs.append(1)
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
        senses=["E"], rhs=[1.0])

"""
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[0][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(a1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[1][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(b1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[2][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(d1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[3][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(e1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[4][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(f1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[5][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(g1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[6][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(h1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
for j in D:
	thevars=[]
	thecoefs=[]
	thevars.append(w[7][j])
	thecoefs.append(1)
	for i in N:
		thevars.append(ii1[i][j])
		thecoefs.append(-1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])
"""

for i in D:
	for j in N:
		if i!=j:
			#print(u[j])
			cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=c[i,j],
		    sense="E",
		    lin_expr=[[u[j],x[i][j]],[1,0]])
for i in N:
	for j in N:
		if i!=j:
			#print(u[j])
			cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=c[i,j],
		    sense="E",
		    lin_expr=[[u[j],u[i]],[1,-1]])
for i in N:
	for j in D:
		if i!=j:
			cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=Q-c[i,j],
		    sense="L",
		    lin_expr=[[u[j],u[i]],[0,1]])

for i in D:
	for j in N:
		cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=loc_x[j]-loc_x[i],
		    sense="E",
		    lin_expr=[[p[j],a[i]],[1,0]])
		
		cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=loc_y[j]-loc_y[i],
		    sense="E",
		    lin_expr=[[o[j],a[i]],[1,0]])
	    
for i in N:
	for j in N:
		if i!=j:
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=loc_x[j]-loc_x[i],
			    sense="E",
			    lin_expr=[[p[j],p[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=loc_y[j]-loc_y[i],
			    sense="E",
			    lin_expr=[[o[j],o[i]],[1,-1]])





i=0
for j in range(len(V)):
	if i !=j:
		cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=1.0,
			    sense="E",
			    lin_expr=[[a[j],a[i]],[1,0]])
	
i=1	
for j in range(len(V)):
	if j>7:
		cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=1.0,
			    sense="E",
			    lin_expr=[[b[j],a[i]],[1,0]])

i=2
for j in range(len(V)):
	if j >7:
		cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=1.0,
			    sense="E",
			    lin_expr=[[d[j],a[i]],[1,0]])

	
i=3
for j in range(len(V)):
	if j>7:
		cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=1.0,
			    sense="E",
			    lin_expr=[[e[j],a[i]],[1,0]])

	
i=4
for j in range(len(V)):
	cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=1.0,
		    sense="E",
		    lin_expr=[[f[j],a[i]],[1,0]])

i=5
for j in range(len(V)):
	if j>7:
		cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=1.0,
			    sense="E",
			    lin_expr=[[g[j],a[i]],[1,0]])

	
i=6
for j in range(len(V)):
	cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=1.0,
		    sense="E",
		    lin_expr=[[h[j],a[i]],[1,0]])

	
i=7
for j in range(len(V)):
	cpx.indicator_constraints.add(
		    indvar=x[i][j],
		    complemented=0,
		    rhs=1.0,
		    sense="E",
		    lin_expr=[[ii[j],a[i]],[1,0]])
	


for i in N:
	for j in N:
		if i!=j:
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[a[j],a[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[b[j],b[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[d[j],d[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[e[j],e[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[f[j],f[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[g[j],g[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[h[j],h[i]],[1,-1]])
			cpx.indicator_constraints.add(
			    indvar=x[i][j],
			    complemented=0,
			    rhs=0.0,
			    sense="E",
			    lin_expr=[[ii[j],ii[i]],[1,-1]])



for i in N:
	for j in D:		
		cpx.indicator_constraints.add(
		indvar=a[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[0][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=b[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[1][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=d[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[2][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=e[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[3][j],x[i][j]],[-1,1]])

		cpx.indicator_constraints.add(
		indvar=f[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[4][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=g[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[5][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=h[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[6][j],x[i][j]],[-1,1]])
		
		cpx.indicator_constraints.add(
		indvar=ii[i],
		complemented=0,
		rhs=0.0,
		sense="L",
		lin_expr=[[w[7][j],x[i][j]],[-1,1]])
		
		




for i in D:
	thevars=[]
	thecoefs=[]
	thevars.append(t[i])
	thecoefs.append(1)
	for j in N:
		thevars.append(x[i][j])
		thevars.append(x[j][i])
		thecoefs.append(-1)
		thecoefs.append(1)
	cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
    senses=["E"], rhs=[0.0])





for i in D:
	for j in range(i+1,len(D)):
		if i !=j:
			cpx.linear_constraints.add(
	    	lin_expr=[([t[i],t[j]],[1,1])],
	    	senses=["L"], rhs=[1.0])
	    	
for i in D:
	for j in range(i+1,len(D)):
		cpx.linear_constraints.add(
	    	lin_expr=[([t[i],t[j]],[1,1])],
	    	senses=["G"], rhs=[-1.0])				    


for i in N:
	cpx.indicator_constraints.add(
		    indvar=a[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][2],x[i][3],x[i][4],x[i][5],x[i][6]],[1,1,1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=b[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][4],x[i][5],x[i][6]],[1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=d[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][0],x[i][4],x[i][5],x[i][6],x[i][7]],[1,1,1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=e[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][0],x[i][6],x[i][7]],[1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=f[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][0],x[i][2],x[i][1],x[i][6],x[i][7]],[1,1,1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=g[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][0],x[i][1],x[i][2]],[1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=h[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][0],x[i][1],x[i][2],x[i][3],x[i][4]],[1,1,1,1,1]])
	cpx.indicator_constraints.add(
		    indvar=ii[i],
		    complemented=0,
		    rhs=0.0,
		    sense="E",
		    lin_expr=[[x[i][2],x[i][3],x[i][4]],[1,1,1]])





































lazycb = cpx.register_callback(LazyCallback)
lazycb.x = x
lazycb.w = w
lazycb.N = N
lazycb.V = V
lazycb.u = u
lazycb.A = A
lazycb.D = D
lazycb.d = d
#lazycb.b1=b1
lazycb.loc_x=loc_x
lazycb.loc_y=loc_y

thevars=[]
thecoefs=[]
####Mip1=[(1, 10), (2, 16), (5, 31), (8, 13), (9, 8), (10, 11), (11, 15), (12, 2), (13, 18), (14, 9), (15, 14), (16, 22), (17, 12), (18, 19), (19, 20), (20, 21), (21, 17), (22, 5), (23, 28), (24, 23), (25, 24), (26, 32), (27, 30), (28, 29), (29, 5), (30, 25), (31, 26), (32, 27)]
#Mip1=[[7,16],[16,17],[17,21],[21,20],[20,7],[7,12],[12,8],[8,9],[9,13],[13,1],[1,10],[10,11],[11,15],[15,14],[14,3],[3,19],[19,18],[18,22],[22,23],[23,3]]
#Mip1=[[7,10],[10,7],[7,8],[8,1],[1,9],[9,3],[3,11],[11,3]]
#Mip1=[[1,9],[9,8],[8,13],[13,18],[18,19],[19,14],[14,10],[10,1],[1,11],[11,12],[12,16],[16,15],[15,20],[20,21],[21,17],[17,22],[22,3],[3,27],[27,32],[32,31],[31,26],[26,25],[25,30],[30,5],[5,29],[29,28],[28,24],[24,23],[23,7]]
#321Mip1=[(1, 10), (3, 27), (5, 29), (8, 13), (9, 8), (10, 11), (11, 12), (12, 17), (13, 18), (14, 9), (15, 14), (16, 15), (17, 16), (18, 19), (19, 20), (20, 21), (21, 22), (22, 3), (23, 5), (24, 23), (25, 30), (26, 25), (27, 26), (28, 24), (29, 28), (30, 31), (31, 32), (32, 5)]
#Mip1=[[7,16],[16,20],[20,21],[21,17],[17,13],[13,12],[12,8],[8,1],[1,9],[9,10],[10,11],[11,3],[3,15],[15,14],[14,18],[18,22],[22,23],[23,19],[19,3]]

#Mip2=[[5, 68], [68, 69], [69, 70], [70, 71], [71, 63], [63, 62], [62, 61], [61, 60], [60, 52], [52, 53], [53, 54], [54, 55], [55, 47], [47, 46], [46, 45], [45, 44], [44, 5], [5, 67], [67, 66], [66, 65], [65, 64], [64, 56], [56, 57], [57, 58], [58, 59], [59, 51], [51, 50], [50, 49], [49, 48], [48, 40], [40, 41], [41, 42], [42, 43], [43, 7], [7, 32], [32, 33], [33, 34], [34, 35], [35, 27], [27, 26], [26, 25], [25, 24], [24, 16], [16, 17], [17, 18], [18, 19], [19, 11], [11, 10], [10, 9], [9, 8], [8, 1], [1, 12], [12, 13], [13, 14], [14, 15], [15, 23], [23, 22], [22, 21], [21, 20], [20, 28], [28, 29], [29, 30], [30, 31], [31, 39], [39, 38], [38, 37], [37, 36], [36, 1]]
#Mip1=[(1, 9), (5, 16), (7, 14), (8, 11), (9, 10), (10, 13), (11, 7), (12, 8), (13, 12), (14, 15), (15, 5), (16, 5)]
#36#Mip1=[[5, 41], [41, 42], [42, 43], [43, 37], [37, 36], [36, 35], [35, 29], [29, 30], [30, 31], [31, 5], [5, 40], [40, 39], [39, 38], [38, 32], [32, 33], [33, 34], [34, 28], [28, 27], [27, 26], [26, 7], [7, 20], [20, 21], [21, 22], [22, 16], [16, 15], [15, 14], [14, 8], [8, 9], [9, 10], [10, 1], [1, 11], [11, 12], [12, 13], [13, 19], [19, 18], [18, 17], [17, 23], [23, 24], [24, 25], [25, 1]]
#49
#Mip1=[[5, 54], [54, 55], [55, 56], [56, 49], [49, 48], [48, 47], [47, 40], [40, 41], [41, 42],[42,5], [5, 53], [53, 52], [52, 51], [51, 50], [50, 43], [43, 44], [44, 45], [45, 46], [46, 39], [39, 38], [38, 37], [37, 36], [36, 7], [7, 29], [29, 30], [30, 31], [31, 32], [32, 25], [25, 24], [24, 23], [23, 22], [22, 15], [15, 16], [16, 17], [17, 18], [18, 11], [11, 10], [10, 9], [9, 8], [8, 1], [1, 12], [12, 13], [13, 14], [14, 21], [21, 20], [20, 19], [19, 26], [26, 27], [27, 28], [28, 35], [35, 34], [34, 33], [33, 1]]
#81
Mip1=[[5, 85], [85, 86], [86, 87], [87, 88], [88, 79], [79, 78], [78, 77], [77, 76], [76, 67], [67, 68], [68, 69], [69, 70], [70, 61], [61, 60], [60, 59], [59, 58], [58, 57], [57, 66], [66, 75], [75, 84], [84, 5], [5, 83], [83, 82], [82, 81], [81, 80], [80, 71], [71, 72], [72, 73], [73, 74], [74, 65], [65, 64], [64, 63], [63, 62], [62, 53], [53, 54], [54, 55], [55, 56], [56, 7], [7, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 39], [39, 38], [38, 37], [37, 36], [36, 35], [35, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 21], [21, 20], [20, 19], [19, 18], [18, 17], [17, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 1], [1, 13], [13, 14], [14, 15], [15, 16], [16, 25], [25, 24], [24, 23], [23, 22], [22, 31], [31, 32], [32, 33], [33, 34], [34, 43], [43, 42], [42, 41], [41, 40], [40, 49], [49, 50], [50, 51], [51, 52], [52, 1]]
for i in Mip1:
	thevars.append(x[i[0]][i[1]])
	thecoefs.append(1)

ee=[]
#ee=[w[5][5],w[5][7],w[7][1],w[1][1]]
aaa=[85, 86, 87, 88, 79, 78, 77, 76, 67, 68, 69, 70, 61, 60, 59, 58, 57, 66, 75, 84,83, 82, 81, 80, 71, 72, 73, 74, 65, 64, 63, 62, 53, 54, 55, 56]
ddd=[44, 45, 46, 47, 48, 39, 38, 37, 36, 35, 26, 27, 28, 29, 30, 21, 20, 19, 18, 17, 8, 9, 10, 11, 12]
ggg=[13, 14, 15, 16, 25, 24, 23, 22, 31, 32, 33, 34, 43, 42, 41, 40, 49, 50, 51, 52]
#see=[w[7][5],w[5][5]]
#aaa=[16]
##ddd=[14,15]
#ggg=[8,9,10,11,12,13]

for i in aaa:
	ee.append(g[i])
for i in ddd:
	ee.append(ii[i])
for i in ggg:
	ee.append(b[i])

#ee=[ii[16],ii[17],ii[21],ii[20],ii1[20][7],ii[12],ii[8],ii[9],ii[13],ii1[13][1],b[10],b[11],b[14],b[15],b1[14][3],e[19],e[18],e[22],e[23],e1[23][3]]
#ee=[ii[10],ii1[10][7],ii[8],ii1[8][1],b[9],b1[9][3],e[11],e1[11][3]]
#ee=[b[9],b[8],b[13],b[18],b[19],b[14],b[10],b[11],b[12],b[16],b[15],b[20],b[21],b[17],b[22],e[27],e[32],e[31],e[26],e[25],e[30],g[29],g[28],g[24],g[23]]
ff=[1]*len(ee)
print(len(ee))
thevars+=ee
thecoefs+=ff
for i in V:
	for j in V:
		if x[i][j] not in thevars:
			thevars.append(x[i][j])
			thecoefs.append(0)
#for i in D:
#	for j in D:
#		if w[i][j] not in thevars:
#			thevars.append(w[i][j])
#			thecoefs.append(0)
for i in V:
	if a[i] not in thevars:
		thevars.append(a[i])
		thecoefs.append(0)
	if b[i] not in thevars:
		thevars.append(b[i])
		thecoefs.append(0)
	if d[i] not in thevars:
		thevars.append(d[i])
		thecoefs.append(0)
	if e[i] not in thevars:
		thevars.append(e[i])
		thecoefs.append(0)
	if f[i] not in thevars:
		thevars.append(f[i])
		thecoefs.append(0)
	if g[i] not in thevars:
		thevars.append(g[i])
		thecoefs.append(0)
	if h[i] not in thevars:
		thevars.append(h[i])
		thecoefs.append(0)
	if ii[i] not in thevars:
		thevars.append(ii[i])
		thecoefs.append(0)



cpx.MIP_starts.add(cplex.SparsePair(thevars,thecoefs),cpx.MIP_starts.effort_level.repair)

cpx.parameters.mip.limits.repairtries=1000
#cpx.parameters.mip.strategy.startalgorithm.set(4)

cpx.parameters.mip.strategy.rinsheur.set(5)     
cpx.write('model5.lp')
cpx.solve()
print(cpx.solution.get_values(x[1][2]))
active_arcs = [r for r in A if cpx.solution.get_values(x[r[0]][r[1]])>0.9]
#print(u[1].solution_value)
plt.scatter(loc_x[1:],loc_y[1:],c='b')
print(active_arcs)
for i,j in active_arcs:
	plt.plot([loc_x[i],loc_x[j]],[loc_y[i],loc_y[j]],c='g')
plt.axis('equal')

print(cpx.solution.get_status_string())
print(cpx.solution.get_objective_value())
for i in D:
	for j in D:
		print(cpx.solution.get_values(w[i][j]),i,j)
#for i in V:
#	print(cpx.solution.get_values(p[i]))
"""
for i in V:
	for j in D:
		print(cpx.solution.get_values(a1[i][j]),i,j)
		

for i in D:
	print(cpx.solution.get_values(t[i]),i)
for j in range(0,8):
	print(j)
	for i in N:
		#print(ii[i])
		print(cpx.solution.get_values(a1[i][j]),cpx.solution.get_values(a[i]),"a",i,j)
		print(cpx.solution.get_values(b1[i][j]),cpx.solution.get_values(b[i]),"b",i,j)
		print(cpx.solution.get_values(d1[i][j]),cpx.solution.get_values(d[i]),"d",i,j)
		print(cpx.solution.get_values(e1[i][j]),cpx.solution.get_values(e[i]),"e",i,j)
		print(cpx.solution.get_values(f1[i][j]),cpx.solution.get_values(f[i]),"f",i,j)
		print(cpx.solution.get_values(g1[i][j]),cpx.solution.get_values(g[i]),"g",i,j)
		print(cpx.solution.get_values(h1[i][j]),cpx.solution.get_values(h[i]),"h",i,j)
		print(cpx.solution.get_values(ii1[i][j]),cpx.solution.get_values(ii[i]),"ii",i,j)
			#print(cpx.solution.get_values(a1[i][j]))


	#for j in D:
	#	print(cpx.solution.get_values(w[i][j]),i,j)
	"""
plt.show()

"""
		print("ss")
		print(cpx.solution.get_values(a1[i][j]))
		print(cpx.solution.get_values(b1[i][j]))
		print(cpx.solution.get_values(c1[i][j]))
		print(cpx.solution.get_values(d1[i][j]))
		print(cpx.solution.get_values(e1[i][j]))
		print(cpx.solution.get_values(f1[i][j]))
		print(cpx.solution.get_values(g1[i][j]))
		print(cpx.solution.get_values(h1[i][j]))
		"""

"""
po = powerset(act)
		po.remove([])
		po.remove(act)
		for i in po:
			thevars=[]
			thecoefs=[]
			pp=0
			for j in i:
				for k in act:
					if k not in i:
						thevars.append(self.x[k][j])
						thevars.append(self.x[j][k])
						thecoefs.append(1)
						thecoefs.append(1)
						pp+=self.get_values(self.x[k][j])+self.get_values(self.x[j][k])
			if pp==0:
				print("sss")
				self.add(constraint=cplex.SparsePair(thevars,thecoefs),
						 sense="G",
						 rhs=1)
"""