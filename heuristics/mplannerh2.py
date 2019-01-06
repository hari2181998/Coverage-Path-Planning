
from __future__ import division
from collections import defaultdict
from math import *
from random import sample
import csv
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


class BaseAlgorithm():

    #def __init__(self):
     #   self.update_data()

    def update_data(self,a,b,c):
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
        self.coords=[[-a/2,-a/2],[-a/2,a/2],[a/2,a/2],[a/2,-a/2]]

        for i in range(1,int(ceil(a/c))):
          self.coords.append([0-a/2,c*i-a/2])
          self.coords.append([c*i-a/2,0-a/2])
          self.coords.append([c*i-a/2,a/2])
          self.coords.append([a/2,c*i-a/2])
        self.denum = len(self.coords)
        print self.coords,self.denum
        #random.shuffle(self.coords)
        for i in range(0,int((a/b))):
          for j in range(0,int((a/b))):
            self.coords.append([-a/2+b/2+(b*j),a/2-b/2-(b*i)])
        #print self.coords,len(self.coords)
        self.cities=range(0,len(self.coords))
        #random.shuffle(self.coords[7:])
        print self.coords,len(self.coords)
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
    def __init__(self,dist,grid,comm,Q):
        self.update_data(dist,grid,comm)
        self.Q=Q
        self.comm=comm
        self.sqr=dist
        self.grid=grid
        ##print self.cities

    def closest_neighbor(self, tour, node, in_tour=False, farthest=False):
        neighbors = self.distances[node]
        ##print node
        ##print neighbors.items()
        ##print tour
        current_dist = [(c, d) for c, d in neighbors.items()
                        if (c in tour)]

        return sorted(current_dist, key=itemgetter(1))[-farthest]


    def add_closest_to_tour(self, tours,tourslength,unass,depot_dist):
        #print tours,tourslength,depot_dist
        best_ratio,best_dist, new_tour = float('inf'),float('inf'), None
        ##print vehlengths
        ##print veh
        ##print tourslength
        ##print tours
        t=1
        tour_index = None
        city1 = None
        c=0.55
        d=0.45
        
        for city in unass:
           ##print city
          for tour in tours:
            a=abs(self.coords[tour[0]][0]-self.coords[city][0])
            b=abs(self.coords[tour[0]][1]-self.coords[city][1])
            x=abs(self.coords[tour[-1]][0]-self.coords[city][0])
            y=abs(self.coords[tour[-1]][1]-self.coords[city][1])
            ##print tour   
            for index in range(len(tour) - 1):
                dist = self.add(tour[index], tour[index + 1], city)
                #print unass
                ##print dist
                ##print vehlengths[tours.index(tour)]
                if len(tour)!=2:
                  ratio = c*dist+d*depot_dist[tours.index(tour)]
                else:
                  ratio = c*(dist+tourslength[tours.index(tour)])+d*depot_dist[tours.index(tour)]
                if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<self.Q and a<=self.comm and b<=self.comm and x<=self.comm and y<=self.comm:
                    best_dist = dist
                    best_ratio = ratio
                    new_tour = tour[:index + 1] + [city] + tour[index + 1:]
                    tour_index = tours.index(tour)
                    city1 = city        
        ##print best_dist                
                    ##print city1               
                                
        return best_dist, new_tour, tour_index,city1

    
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
                      if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<self.Q:
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

    def depot_array(self,denum,unass):
      depot=[]
      x=self.comm
      for i in range(0,len(unass)):
        depot.append([])
     
      for i in range(0,denum):
        for f in range(0,len(unass)):
          k=0
          for j in unass[f]:
            if abs(self.coords[i][0]-self.coords[j][0])<=x and abs(self.coords[i][1]-self.coords[j][1])<=x:
              k+=1
              print(j,i,"ss",k)
          if k==len(unass[f]):
            depot[f].append(i)
      return depot

    def depot_path(self,depot1,depot2,depot3,depot4):
      depot_path=[]
      depot_insec=[]
      depot=[depot1,depot2,depot3,depot4]
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
    def intersection(self,a,b):
      for i in a:
        for j in b:
          if i==j:
            x=i 

      return x 

    



    
    def cheapest_insertion(self):
      denum=self.denum
      comm=self.comm
      dist=self.sqr
      grid=self.grid
      #c=1
      g=None
      ##print a
      #tour = [0,a]
      c=[]
      d=[]
      depot_dist=[]
      gr=4
      depot_insec=[]
      total_len=0
      prevcdep=None
      same_y=[]
      l=len(self.coords)
      tourslength=[]
      tours=[]
      unass2=[]      
      #tourslength.append(0)
      qqq=[]
      ppp= self.cities[denum:l]
      for i in range(0,int (ceil(dist/comm)**2)):
        unass2.append([])
      
      
      for i in range(0,int(ceil(dist/comm))):
        if i%2==0:
          for j in range(0,int(ceil(dist/comm))):
            qqq.append([i,j])
        else:
          for j in range(0,int (ceil(dist/comm))):
            qqq.append([i,int (ceil(dist/comm))-j-1])
      print qqq

      for k in self.cities[denum:l]:
        a=self.coords[k][0]
        b=self.coords[k][1]
        #print k,a,b
        for i in range(0,int(ceil(dist/comm))):
          for j in range(0,int(ceil(dist/comm))):
            print k,a,b,comm*i-(dist/2), comm*(i+1)-(dist/2)
            if b>=comm*i-(dist/2) and b<=comm*(i+1)-(dist/2):
              if a>=comm*j-(dist/2) and a<=comm*(j+1)-(dist/2):
                if k in ppp:
                  #print "s"
                  unass2[qqq.index([i,j])].append(k)
                  ppp.remove(k)
      print unass2
      print ppp
          #print unass4,"4"
      #unass2=[unass1,unass22,unass3,unass4]
      depot=self.depot_array(denum,unass2)
      depot1=depot[0]
      depot2=depot[1]
      depot3=depot[2]
      depot4=depot[3]
      #depot=[depot1,depot2,depot3,depot4]
      print depot
      depot_insec.append(self.intersection(depot1,depot2))
      depot_insec.append(self.intersection(depot2,depot3))
      depot_insec.append(self.intersection(depot3,depot4))
      depot_insec.append(self.intersection(depot4,depot1))
      dep=[]
      
      for i in depot:
        dep.append([])
        for j in i:
          for k in i:
            dep[-1].append([j,k])
            tours.append([j,k])
            c.append(self.distances[k][depot_insec[depot.index(i)]])
            d.append(self.distances[k][depot_insec[depot.index(i)-1]])
            tourslength.append(self.distances[j][k])
      unass=self.cities[denum:l]
      tours2=tours[:]
      for i in range(0,len(tours)):
        depot_dist.append(min(c[i],d[i]))
      f=0






      while len(unass)!=0:

        length, tour, index, city1 = self.add_closest_to_tour(tours,tourslength,unass,depot_dist)
        print tours,depot_dist,index,len(unass)



        if f==0 :

          for i in depot:
            if tour[0] in i and tour[-1] in i:
              x=depot_insec[depot.index(i)]
              y=depot_insec[depot.index(i)-1]
              break

          if tour[-1] not in depot_insec:
            
            if c[index]<d[index]:
              depot_dist=c[:]
              #print c
              p=x
              g=True
            else:
              depot_dist=d[:]
              p=y 
              g=False

            tours[index]=tour
            tourslength[index]+=length
            unass.remove(city1)
            a=tourslength
            o=depot_dist
            ##print tours
            i=0
            while i<len(tours):
              tour2=tours[i]
              ##print tour2
              i+=1
              if (len(tour2)==2):
                del depot_dist[tours.index(tour2)]
                #print c
                tourslength.remove(a[tours.index(tour2)])
                tours.remove(tour2)
                i-=1
            #print c
            #print d
            if c[index]<d[index]:
              
              p=x
              tours.append([tour[-1],p])
              depot_dist.append(c[tours2.index(tours[-1])])
              tourslength.append(self.distances[tour[-1]][p])
              tours.append([tour[-1],tour[-1]])
              tourslength.append(self.distances[tour[-1]][tour[-1]])
              depot_dist.append(c[tours2.index(tours[-1])])

            else:
              
              p=y
              tours.append([tour[-1],p])
              tourslength.append(self.distances[tour[-1]][p])
              depot_dist.append(d[tours2.index(tours[-1])])
              tours.append([tour[-1],tour[-1]])
              tourslength.append(self.distances[tour[-1]][tour[-1]])
              depot_dist.append(d[tours2.index(tours[-1])])



          else:

            for i in unass2:
              if tour[1] in i:
                unindex=unass2.index(i)
            #print unindex,unass2[unindex]
            if unindex==depot_insec.index(tour[-1]):
              depot_dist=c[:]
              g = True
              p=x
            else:
              depot_dist=d[:]
              g = False
              p=y
            tours[index]=tour
            tourslength[index]+=length
            unass.remove(city1)
            a=tourslength
            o=depot_dist
            ##print tours
            i=0
            while i<len(tours):
              tour2=tours[i]
              ##print tour2
              i+=1
              if (len(tour2)==2):
                del depot_dist[tours.index(tour2)]
                #print c
                tourslength.remove(a[tours.index(tour2)])
                tours.remove(tour2)
                i-=1
                """
            if g is True:
              
              p=x
              tours.append([tour[-1],p])
              depot_dist.append(c[tours2.index(tours[-1])])
              tourslength.append(self.distances[tour[-1]][p])
              tours.append([tour[-1],tour[-1]])
              tourslength.append(self.distances[tour[-1]][tour[-1]])
              depot_dist.append(c[tours2.index(tours[-1])])

            else:
              
              p=y
              tours.append([tour[-1],p])
              tourslength.append(self.distances[tour[-1]][p])
              depot_dist.append(d[tours2.index(tours[-1])])
              tours.append([tour[-1],tour[-1]])
              tourslength.append(self.distances[tour[-1]][tour[-1]])
              depot_dist.append(d[tours2.index(tours[-1])])
            #print depot_dist
            """
            
              

            if g is True:
              for i in range(0,len(depot_dist)):
                depot_dist[i]=self.distances[tours[i][-1]][depot_insec[unindex+1]]

              j=tour[-1]
              for k in depot[unindex+1]:
                tours.append([j,k])
                tourslength.append(self.distances[j][k])
                depot_dist.append(self.distances[k][depot_insec[unindex+1]])
            if g is False:
              for i in range(0,len(depot_dist)):
                depot_dist[i]=self.distances[tours[i][-1]][depot_insec[unindex-2]]
              j=tour[-1]
              for k in depot[unindex-1]:
                tours.append([j,k])
                tourslength.append(self.distances[j][k])
                depot_dist.append(self.distances[k][depot_insec[unindex-2]])



          #print depot_dist,tours,g
        if index!=None and f>0 :
          #print tours
          tours[index]=tour
          tourslength[index]+=length
          if len(tour)==3:
            if tours[-1]!=tour:
              tours.insert(index+1,[tour[0],tour[-1]])
              depot_dist.insert(index+1,depot_dist[index])
              tourslength.insert(index+1,self.distances[tour[0]][tour[-1]])
            c=tour[0]
            d=tour[-1]
            print c,d

            for j in depot:
              if c in j and d in j :
                u=depot.index(j)
                break
            print depot[u]
            for i in depot[u]:
                  
                  if [c,i] in tours and i!=c:
                    
                    del tourslength[tours.index([c,i])]
                    del depot_dist[tours.index([c,i])]
                    #print vehlengths
                    tours.remove([c,i])

            
          unass.remove(city1)
          jj=0
        if index==None:
          jj+=1
          if jj==2:
            break
          a=tourslength
          o=depot_dist
          ##print tours
          i=0
          while i<len(tours):
            tour2=tours[i]
            ##print tour2
            i+=1
            if (len(tour2)==2):
              del depot_dist[tours.index(tour2)]
              tourslength.remove(a[tours.index(tour2)])
              tours.remove(tour2)
              i-=1
          if g is True:
            for i in tours:
              if i[0]!=i[-1]:
                b=i[-1]
            for h in depot_insec:
              if b==h:
                q=depot_insec.index(h)
                break
            f=q+1
            if q==3:
              f=0
            for k in depot[f]:
              tours.append([b,k])
              tourslength.append(self.distances[k][b])
              depot_dist.append(self.distances[k][depot_insec[f]])
          if g is False:
            for i in tours:
              if i[0]!=i[-1]:
                b=i[-1]
            for h in depot_insec:
              if b==h:
                q=depot_insec.index(h)
                break
            for k in depot[q]:
              tours.append([b,k])
              tourslength.append(self.distances[k][b])
              depot_dist.append(self.distances[k][depot_insec[q-1]])
        f+=2
      for i in tours:
        if len(i)==2:
          del tourslength[tours.index(i)]
          tours.remove(i)
      total=0
      totaltour=0
      for i in range(0,len(tourslength)):
        totaltour+=tourslength[i]
      total=totaltour
      for i in range(0,len(tours)-1):
        if tours[i+1][0]==tours[i][-1]:
          total+=self.perp_dist(tours[i][0],tours[i][-1])
      xx=[]
      yy=[]
      for i in self.cities:
        xx.append(self.coords[i][0]+dist/2)
        yy.append(self.coords[i][1]+dist/2)



      return tours,tourslength,totaltour,total,depot_dist,xx,yy




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
      for i in range(0,len(self.coords)):
        x.append(self.coords[i][0])
        y.append(self.coords[i][1])
        plt.plot(self.coords[i][0],self.coords[i][1],'rs')
      
      #plt.show() 
#r= BaseAlgorithm()


xxx= 'QGC WPL 110\r\n'
import utm
#from math import *
import numpy as np
file = open("mission.txt","r")
a=file.readlines()
file.close()
lat=[]
lon=[]
#xx+=a[2]
if a[1][1]=='\t':
  print "s"
j=0
print a
index=None
for k in a:
  if a.index(k)!=0:
    j=0
    lat1='s'
    lon1='s'
    for i in range (0,len(k)):
      if k[i]=='\t':
        j+=1
        print j
      if j==8:
        index=i
        break
    for i in range(index+1,len(k)):
      if k[i]=='\t':
        index=i
        break
      lat1+=k[i]
    for i in range(index+1,len(k)):
      if k[i]=='\t':
        #index=i
        break
      lon1+=k[i] 

    print k
    print index
    lat.append(float(lat1[1:]))
    lon.append(float(lon1[1:]))
print lat
print lon
e2,n2,aa,bb = utm.from_latlon(lat[1],lon[1])
e1,n1,_,_ = utm.from_latlon(lat[0],lon[0])
angle= atan2(n2-n1,e2-e1)
dist=np.hypot(e2-e1,n2-n1)

def takeoff(lat,lon):
  return '\t0\t3\t22\t0\t5\t0\t0\t' + str(lat) +'\t'+ str(lon) + '\t20\t1\r\n'
def waypoint(lat,lon):
  return '\t0\t3\t16\t0\t5\t0\t0\t' + str(lat) +'\t'+ str(lon) + '\t20\t1\r\n'
def land(lat,lon):
  return '\t0\t3\t21\t0\t5\t0\t0\t' + str(lat) +'\t'+ str(lon) + '\t20\t1\r\n'

def utm1(x,y,e1,n1,angle):
  x1=x*cos(angle)-y*sin(angle)
  y1=x*sin(angle)+y*cos(angle)
  x1+=e1
  y1+=n1
  #print x1,y1
  lat,lon= utm.to_latlon(x1,y1,aa,bb)
  return lat,lon
















print dist































x= TourConstructionHeuristics(dist=dist,grid=20,comm=dist/2,Q=300)
tours,lengths,totaltour,total,depot_dist,xx,yy=x.cheapest_insertion()
print tours
print lengths
print totaltour
print total
print depot_dist
x.plot(tours)
plt.show()

totaltour=[totaltour]
veh=[tours]


tours1, lengths1,veh1,vehlength1 = x.samedis(tours,lengths,veh,totaltour)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,1):
  x.plot(veh1[i])

plt.show()
tours1, lengths1,veh1,vehlength1 = x.samedis(tours1,lengths1,veh1,vehlength1)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,1):
  x.plot(veh1[i])

plt.show()
k=0
for i in tours1:
  for j in range(0,len(i)):
    if j==0:
      lat,lon=utm1(xx[i[j]],yy[i[j]],e1,n1,angle)
      xxx+=str(k)+takeoff(lat,lon)
      k+=1
    elif j== len(i)-1:
      lat,lon=utm1(xx[i[j]],yy[i[j]],e1,n1,angle)
      xxx+=str(k)+land(lat,lon)
      k+=1
    else:
      lat,lon=utm1(xx[i[j]],yy[i[j]],e1,n1,angle)
      xxx+=str(k)+waypoint(lat,lon)
      k+=1

file=open("mission1.txt","w")
file.write(xxx)
file.close()


"""
tours,lengths,total=x.greedy2()
print tours
print lengths
print total
total=[total]
veh=[tours]
x.plot(tours)
plt.show()

tours1, lengths1,veh1,vehlength1 = x.samedis(tours,lengths,veh,total)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,1):
  x.plot(veh1[i])

plt.show()
"""



"""mission planer
qground control
dronecode
for i in range(0,denum):
        for j in range(0,denum):
          if ([i,j]in tours or [j,i]in tours):
            continue
          else:
            tours.append([i,j])
            tourslength.append(self.distances[i][j])
            vehlengths.append(0)"""
