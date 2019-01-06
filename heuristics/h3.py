

from collections import defaultdict
from math import asin, cos, radians, sin, sqrt
from random import sample
import csv
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


class BaseAlgorithm():

    #def __init__(self):
     #   self.update_data()

    def update_data(self,a,b):
        filename = "data3.csv"
        """
        self.cities = []
        #self.size = len(self.cities)
        self.coords = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader: 
                self.coords.append([float(row[0]),float(row[1])])
        self.cities = range(0,len(self.coords))
        """
        self.cities=[]
        #self.coords=[[-a/2,a/2],[-a/2,0],[0,a/2],[0,-a/2],[a/2,a/2],[a/2,-a/2],[a/2,0],[-a/2,-a/2]]
        self.coords=[[-a/2,a/2],[0,a/2],[a/2,a/2],[a/2,0],[a/2,-a/2],[0,-a/2],[-a/2,-a/2],[-a/2,0]]
        #random.shuffle(self.coords)
        for i in range(0,int(a/b)):
          for j in range(0,int(a/b)):
            self.coords.append([-a/2+b/2+(b*j),a/2-b/2-(b*i)])
        #print self.coords,len(self.coords)
        self.cities=range(0,len(self.coords))
        #random.shuffle(self.coords[7:])
        print self.coords
        self.size = len(self.cities)
        self.distances = self.compute_distances()

   
    def haversine_distance(self, cityA, cityB):
        coord1 = self.coords[cityA] 
        coord2= self.coords[cityB]
        a = (coord1[0]-coord2[0])**2+(coord1[1]-coord2[1])**2
        c = sqrt(a)
        return c

    def compute_distances(self):
        self.distances = defaultdict(dict)
        for cityA in self.cities:
            for cityB in self.cities:
                if cityB not in self.distances[cityA]:
                    distance = self.haversine_distance(cityA, cityB)
                    self.distances[cityA][cityB] = distance
                    self.distances[cityB][cityA] = distance
        return self.distances

    # add node k between node i and node j
    def add(self, i, j, k):
        return self.distances[i][k] + self.distances[k][j] - self.distances[i][j]

   


class TourConstructionHeuristics(BaseAlgorithm):
     
    # find the neighbor k closest to the tour, i.e such that
    # cik + ckj - cij is minimized with (i, j) an edge of the tour
    # add k between the edge (i, j), resulting in a tour with subtour (i, k, j)
    # used for the cheapest insertion algorithm
    def __init__(self):
        self.update_data(40,5.5)
        ##print self.cities

    def closest_neighbor(self, tour, node, in_tour=False, farthest=False):
        neighbors = self.distances[node]
        ##print node
        ##print neighbors.items()
        ##print tour
        current_dist = [(c, d) for c, d in neighbors.items()
                        if (c in tour)]

        return sorted(current_dist, key=itemgetter(1))[-farthest]


    def add_closest_to_tour(self, tours,tourslength,unass,veh,vehlengths):
        best_ratio,best_dist, new_tour = float('inf'),float('inf'), None
        ##print vehlengths
        ##print veh
        ##print tourslength
        ##print tours
        t=1
        tour_index = 0
        city1 = 0
        c=0.3
        d=0.7
        vehi=vehindex=None
        for city in unass:
           ##print city
          for tour in tours:
            ##print tour   
            for index in range(len(tour) - 1):
                dist = self.add(tour[index], tour[index + 1], city)
                #print unass
                ##print dist
                ##print vehlengths[tours.index(tour)]
                if len(tour)!=2:
                  ratio = c*dist+d*(vehlengths[tours.index(tour)]+dist)
                else:
                  ratio = c*dist+d*(vehlengths[tours.index(tour)]+dist+tourslength[tours.index(tour)])
                if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<150:
                    best_dist = dist
                    best_ratio = ratio
                    new_tour = tour[:index + 1] + [city] + tour[index + 1:]
                    tour_index = tours.index(tour)
                    city1 = city
        for p in range(0,t):
          if tours[tour_index] in veh[p]:
            
            vehi=p
            vehindex = veh[p].index(tours[tour_index])

        ##print best_dist                
                    ##print city1
               
                                
        return best_dist, new_tour, tour_index,city1,vehi,vehindex

    
        ##print unass


    def samedis(self,tours,tourslength,veh,vehlength):
      c=0.5
      d=0.5
      t=1
      for tour1 in tours:
        if len(tour1)!=2:                 
          i=1   
          while (i<len(tour1)-1):
              ##print(len(tour1))
              ##print("!@#!")
              for j in range(0,t):
                if tour1 in veh[j]:
                  o=j
                  p=veh[j].index(tour1)
                  b=vehlength[j]
              ##print veh    ##print b,"s" 
              best_dist = self.add(tour1[i-1], tour1[i+1], tour1[i])
              h=best_dist
              best_ratio = c*best_dist + d*(b)
              
                #print best_dist,best_ratio,"sss"
              #print "ddd"
              ##print("!!!!")
              
              for tour in tours:
                  for j in range(0,t):
                    if tour in veh[j]:
                      a=vehlength[j]
                      w=j
                      s=veh[j].index(tour)
                  ##print("******")

                  if tour != tour1 and len(tour)!=2 :
                    ##print a
                    for index in range(len(tour) - 1):
                      ##print tour
                      ##print index
                      dist = self.add(tour[index], tour[index + 1], tour1[i])
                      #print dist
                      ##print dist
                      ratio = c*dist + d*(a+dist)
                      if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<150:
                        best_dist = dist
                        w1=w
                        s1=s
                        new_tour = tour[:index + 1] + [tour1[i]] + tour[index + 1:]                        
                        tour_index = tours.index(tour)
                        best_ratio = c*best_dist + d*(a+dist)
                        
                          #print best_dist,"fff"
                          #print new_tour,best_ratio
                          #print c*best_dist + d*(b)

              if best_ratio < c*h + d*(b):
                  ##print veh
                  tours[tour_index]=new_tour
                  
                    #print tours[tour_index]
                  tourslength[tour_index]+= best_dist
                  veh[w1][s1]=new_tour
                  vehlength[w1]+=best_dist
                  ##print veh 
                  tourslength[tours.index(tour1)]-=self.add(tour1[i-1], tour1[i + 1], tour1[i])
                  
                  ##print o,i
                  ##print vehlength[o]
                  vehlength[o]-=self.add(tour1[i-1], tour1[i + 1], tour1[i])
                  #print veh
                  #print tour1
                  veh[o][p].remove(tour1[i])
                  #print tour1
                  #print veh
                  if (len(tour1)==2):
                    vehlength[o]-=self.distances[tour1[0]][tour1[1]]
                  #tour1.remove(tour1[i])
                  #print veh
              else:
                  i+=1             
      

      ##print self.distances                ##print(i) 
      return tours, tourslength,veh,vehlength
    def perp_dist(self,a,b):
      dist = abs(self.coords[a][0]-self.coords[b][0])+abs(self.coords[a][1]-self.coords[b][1])
      return dist
    def depot_array(self,denum,unass1,unass2,unass3,unass4):
      depot1=[]
      depot2=[]
      depot3=[]
      depot4=[]
      for i in range(0,denum):
        k=0
        for j in unass1:
          
          
          if abs(self.coords[i][0]-self.coords[j][0])<=20 and abs(self.coords[i][1]-self.coords[j][1])<=20:
            k+=1
            print(j,i,"ss",k)
        if k==len(unass1):
          depot1.append(i)
        k=0 #print i ,self.coords[unass1[0]][0]
        for j in unass2:
          
          if abs(self.coords[i][0]-self.coords[j][0])<=20 and abs(self.coords[i][1]-self.coords[j][1])<=20:
            k+=1
        if k==len(unass2):
          depot2.append(i)
        k=0  #print i ,self.coords[unass2[0]][0]
        for j in unass3:
          
          if abs(self.coords[i][0]-self.coords[j][0])<=20 and abs(self.coords[i][1]-self.coords[j][1])<=20:
            k+=1
        if k==len(unass3):
          depot3.append(i)
        k=0  #print i ,self.coords[unass3[0]][0]
        for j in unass4:
          
          if abs(self.coords[i][0]-self.coords[j][0])<=20 and abs(self.coords[i][1]-self.coords[j][1])<=20:
            k+=1
        if k==len(unass4):
          depot4.append(i)
      return depot1,depot2,depot3,depot4

    def depot_path(self,depot1,depot2,depot3,depot4):
      depot_path=[]
      depot_insec=[]
      g=None
      depot=[depot1,depot2,depot3,depot4]
      print(depot)
      for i in range(0,len(depot)-1):
        b=float('inf')
        for j in range(0,len(depot[i])):
          for k in range(0,len(depot[i+1])):
            a=self.perp_dist(depot[i][j],depot[i+1][k])
            if a==0:
              g=depot[i][j]
            elif a<b:
              b=a
              f=depot[i][j]
          if f==g:
            b=float('inf')
            f= None

        
        if i==0:
          depot_path.append(g)
          depot_insec.append(g)
        if i!=0:
          depot_path.append(f)
          depot_path.append(g)
          depot_insec.append(g)
      return depot_path,depot_insec


    def closest_city(self,unass,city):
      best_dist=float('inf')
      closest=None
      for x in unass:
        dist=self.distances[city][x]
        #print dist,x
        if dist<best_dist:
          best_dist=dist
          #print "s"
          closest=x
      return closest,best_dist

    def greedy3(self,denum,c,d,a,tourslength,tours,unass,depot_path):
      m=0
      
      same_y=[]
      while len(unass)!=0:
        print tours
        if len(tours[-1])==1:
          x,best_dist=self.closest_city(unass,tours[-1][0])
          #print x
          tours[-1].append(x)
          unass.remove(x)
          tourslength[-1]+=best_dist
          if m is 0:
            y=self.coords[x][0]
          m+=1
        else:
          a=self.coords[tours[-1][-1]][0]
          b=self.coords[tours[-1][-1]][1]
          best_dist=float('inf')
          city1=None
          for city in unass:
            if self.coords[city][1]==b:
              same_y.append(city)
          for i in range(0,len(same_y)):
            for  j in range(i+1,len(same_y)):
              if self.distances[same_y[j]][tours[-1][-1]]<self.distances[same_y[i]][tours[-1][-1]]:
                a=same_y[i]
                b=same_y[j]
                same_y[i]=b
                same_y[j]=a
          print same_y
          #print same_y,unass
          while len(same_y)!=0:
            #print same_y
            """
            best_dist=float('inf')
            city1=None
            for city in same_y:
              dist=self.distances[city][tours[-1][-1]]
              #print "s",tours[-1],city,dist
              if dist<best_dist:
                city1=city
                best_dist=dist
            """
            best_dist=self.distances[same_y[0]][tours[-1][-1]]
            city1=same_y[0]
            best_dep=float('inf')
            best_ratio=float('inf')
            cdep=None
            for i in depot_path:
              mindep=abs(self.coords[tours[-1][0]][0]-self.coords[i][0])+abs(self.coords[tours[-1][0]][1]-self.coords[i][1])
              dep=self.distances[city1][i]
              ratio = c*dep+d*mindep          
              if ratio<best_ratio:
                best_dep=dep
                cdep=i
                best_ratio=ratio
            if self.coords[city1][0]==y and tourslength[-1]>108:
              f=True
            else:
              f= False  
            #print f  
            
            if best_dist+tourslength[-1]+best_dep<150:
              #print city1
              tours[-1].append(city1)
              unass.remove(city1)
              same_y.remove(city1)
              tourslength[-1]+=best_dist
            else:
              tours[-1].append(prevcdep)
              tourslength[-1]+=prevbest_dep
              tours.append([prevcdep])
              tourslength.append(0)
            if f==True:
              tours[-1].append(cdep)
              tourslength[-1]+=best_dep
              tours.append([cdep])
              tourslength.append(0)
           
            prevcdep=cdep
            prev_bestdep=best_dep
          if city1==None:
            for city in unass:
              if self.coords[city][0]==a:
                dist=self.distances[city][tours[-1][-1]]
                if dist<best_dist:
                  city1=city
                  best_dist=dist
            if city1==None:
               city1,best_dist=self.closest_city(unass,tours[-1][0])
            best_dep=float('inf')
            best_ratio=float('inf')
            cdep=None
            #print(city1)
            for i in depot_path:
              mindep=abs(self.coords[tours[-1][0]][0]-self.coords[i][0])+abs(self.coords[tours[-1][0]][1]-self.coords[i][1])
              print(city1,i)
              dep=self.distances[city1][i]
              ratio = c*dep+d*mindep          
              if ratio<best_ratio:
                best_dep=dep
                cdep=i
                best_ratio=ratio
            if self.coords[city1][0]==y  and tourslength[-1]>108:
              f=True
            else:
              f= False 
                
            #print f,self.coords[city1][0]
            if best_dist+tourslength[-1]+best_dep<150:
              tours[-1].append(city1)
              unass.remove(city1)
              tourslength[-1]+=best_dist
            else:
              tours[-1].append(prevcdep)
              tourslength[-1]+=prevbest_dep
              tours.append([prevcdep])
              tourslength.append(0)
            if f==True:
              tours[-1].append(cdep)
              tourslength[-1]+=best_dep
              tours.append([cdep])
              tourslength.append(0)
            prevcdep=cdep
            prevbest_dep=best_dep
          
      return tours,tourslength






    def greedy2(self):
      denum=8
      c=1
      d=0
      gr=4
      total_len=0
      prevcdep=None
      same_y=[]
      l=len(self.coords)
      tourslength=[]
      tours=[]      
      tourslength.append(0)
      unass = self.cities[denum:l]
      unass1=[]
      unass2=[]
      unass3=[]
      unass4=[]
      for i in range(denum,l):
        a=self.coords[i][0]
        b=self.coords[i][1]
        #print a,b,i
        if a>0 and b>0:
          unass4.append(i)
          #print unass3,"3"
        elif a<=0 and b>=0:
          unass3.append(i)
          #print unass2,"2"
        elif a<0 and b<0:
          unass2.append(i)
          #print unass1,"1"
        elif a>=0 and b<=0:
          unass1.append(i)
          #print unass4,"4"
      print(unass1,unass2,unass3,unass4)
      depot1,depot2,depot3,depot4=self.depot_array(denum,unass1,unass2,unass3,unass4)
      depot_path,depot_insec=self.depot_path(depot1,depot2,depot3,depot4)
      print depot_path,"s"
      print depot1,depot2,depot3,depot4
      print depot_insec
      f=None
      tour=[depot_path[0]]
      tours.append(tour)
      tours,tourslength=self.greedy3(denum,c,d,l,tourslength,tours,unass1,depot_path)
      print tourslength,tours,self.distances[tours[-1][-1]][depot_insec[0]]      
      tourslength[-1]+=self.distances[tours[-1][-1]][depot_insec[0]]
      tours[-1].append(depot_insec[0])
      tours.append([depot_insec[0]])
      tourslength.append(0)
      tours,tourslength=self.greedy3(denum,c,d,l,tourslength,tours,unass2,depot_path)
      print tourslength,tours
      tourslength[-1]+=self.distances[tours[-1][-1]][depot_insec[1]]
      tours[-1].append(depot_insec[1])      
      tours.append([depot_insec[1]])
      tourslength.append(0)
      tours,tourslength=self.greedy3(denum,c,d,l,tourslength,tours,unass3,depot_path)
      print tourslength,tours
      tourslength[-1]+=self.distances[tours[-1][-1]][depot_insec[2]]
      tours[-1].append(depot_insec[2])      
      tours.append([depot_insec[2]])
      tourslength.append(0)
      tours,tourslength=self.greedy3(denum,c,d,l,tourslength,tours,unass4,depot_path)
      print tourslength,tours
      best_dep=float('inf')
      cdep=None
      for i in range(0,denum):
        dep=self.distances[i][tours[-1][-1]]
        if dep<best_dep:
          best_dep=dep
          cdep=i
      tours[-1].append(cdep)
      tourslength[-1]+=best_dep






      for i in range(0,len(tourslength)):
        total_len+=tourslength[i]
      total=total_len
      for i in range(0,len(tours)-1):
        total+=self.perp_dist(tours[i][0],tours[i][-1])
      return tours,tourslength,total_len,total





    def plot (self,tours):
      b = ['r','b','g','c']
      j=0      
      for tour in tours:
        if len(tour)!=2:                
          for i in range (0,len(tour)-1):
            if i != len(self.coords)-1:
              plt.plot([self.coords[tour[i]][0],  self.coords[tour[i+1]][0]],[self.coords[tour[i]][1],self.coords[tour[i+1]][1]], b[j])  
              #plt.show(block=False)
        if j<3:
          j+=1   
        else:
          j=0
      
      x=[]
      y=[]
      c=['bs','rs','gs','cs','ms']
      for i in range(0,5):
        x.append(self.coords[i][0])
        y.append(self.coords[i][1])
        plt.plot(self.coords[i][0],self.coords[i][1],c[i])
      
      #plt.show() 
#r= BaseAlgorithm()
x= TourConstructionHeuristics()
tours,lengths,total_len,total=x.greedy2()
print tours
print lengths
print total
print total_len
total_len=[total_len]
veh=[tours]
x.plot(tours)
plt.show()

tours1, lengths1,veh1,vehlength1 = x.samedis(tours,lengths,veh,total_len)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,1):
  x.plot(veh1[i])

plt.show()




"""mission planer
qground control
dronecode"""