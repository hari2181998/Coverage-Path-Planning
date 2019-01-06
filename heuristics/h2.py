

from collections import defaultdict
from math import asin, cos, radians, sin, sqrt
from random import sample
import csv
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt


class BaseAlgorithm():

    #def __init__(self):
     #   self.update_data()

    def update_data(self):
        filename = "data.csv"
        self.cities = []
        #self.size = len(self.cities)
        self.coords = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader: 
                self.coords.append([float(row[0]),float(row[1])])
        self.cities = range(0,len(self.coords))
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
        self.update_data()
        #print self.cities

    def closest_neighbor(self, tour, node, in_tour=False, farthest=False):
        neighbors = self.distances[node]
        #print node
        #print neighbors.items()
        #print tour
        current_dist = [(c, d) for c, d in neighbors.items()
                        if (c in tour)]

        return sorted(current_dist, key=itemgetter(1))[-farthest]


    def add_closest_to_tour(self, tours,tourslength,unass,veh,vehlengths):
        best_ratio,best_dist, new_tour = float('inf'),float('inf'), None
        #print vehlengths
        #print veh
        #print tourslength
        #print tours
        t=3
        tour_index = 0
        city1 = 0
        c=0.5
        d=0.5
        vehi=vehindex=None
        for city in unass:
           #print city
          for tour in tours:
            #print tour   
            for index in range(len(tour) - 1):
                dist = self.add(tour[index], tour[index + 1], city)
                #print dist
                #print vehlengths[tours.index(tour)]
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

        #print best_dist                
                    #print city1
               
                                
        return best_dist, new_tour, tour_index,city1,vehi,vehindex

    def nearest_insertion(self, farthest=True):
      a = len(self.coords)-1
      tour = [0,a]
      tours = [tour]
      tourslength = [0]
      cantass=[[]]
      unass = self.cities[1:a]
      #print unass
      while len(unass) != 0:
          best, best_len = None, 0 if farthest else float('inf')
          # (selection step) given a sub-tour,we find node r not in the
          # sub-tour closest to any node j in the sub-tour,
          # i.e. with minimal c_rj
          for tour in tours:
              for city in unass:
                print cantass
                print tour
                if city in cantass[tours.index(tour)]:
                  continue
                # we consider only the distances to nodes already in the tour
                print tour,city
                _, length = self.closest_neighbor(tour, city, True)
                if (length > best_len if farthest else length < best_len):
                    city1, best_len,tour1 = city, length,tour
          #print city1
          # (insertion step) we find the arc (i, j) in the sub-tour which
          # minimizes cir + crj - cij, and we insert r between i and j

          best_dist, new_tour = float('inf'), None
        
        
          for index in range(len(tour1) - 1):
              dist = self.add(tour1[index], tour1[index + 1], city1)
              #print dist
              if dist < best_dist and (tourslength[tours.index(tour1)]+dist)<150:
                  best_dist = dist
                  new_tour = tour1[:index + 1] + [city1] + tour1[index + 1:]
                  tour_index = tours.index(tour1)
          if best_dist == float('inf'):
            cantass[tours.index(tour1)].append(city1)
                  
          if tour_index is (len(tours)-1):
             tour2=[0,a]
             tours.append(tour2)
             cantass.append([])
             tourslength.append(0)
          #print tours
          if city1 != 0 and new_tour != None:
             unass.remove(city1)
             print unass
             tourslength[tour_index] += best_dist
             tours[tour_index]=new_tour
        #print unass
        #print self.cities
        #return tours,tourslength
                     

    
    def cheapest_insertion(self):
        denum=5
        t=3
        v=0
        a= len(self.coords)-denum
        #tour = [0,a]
        tours = []
        tourslength = []
        vehlengths=[]
        for i in range(0,denum):
          for j in range(0,denum):
            if ([i,j]in tours or [j,i]in tours):
              continue
            else:
              tours.append([i,j])
              tourslength.append(self.distances[i][j])
              vehlengths.append(0)
        # we find the closest node R to the first node
        unass = self.cities[denum:a]
        veh = [[],[],[],[],[],[]]
        while len(unass) != 0:
            length, tour, index, city1,vehi,vehindex = self.add_closest_to_tour(tours,tourslength,unass,veh,vehlengths)
            print tours
            print tourslength
            #print tour
            #print veh
            #print vehi
            #print vehindex


            if city1 != 0 and tour != 0:
              unass.remove(city1)
              tourslength[index] += length
              tours[index]=tour
            if vehi!=None:
              veh[vehi][vehindex]=tour
              x=vehlengths[tours.index(tour)]
              for j in range(0,len(tours)):
                    
                    if vehlengths[j]==x:
                      #print"s"
                      vehlengths[j]+=length
            if len(tours[index]) is 3:
              v+=1
              
              if v<t:
                tour1=[tour[0],tour[2]]
                tours.append(tour1)
                tourslength.append(self.distances[tour[0]][tour[2]])
                veh[v-1].append(tour)
                vehlengths.append(0)
                vehlengths[index]=tourslength[index]
                #print veh
                #print veh
                
              if v==t:
                veh[v-1].append(tour)
                vehlengths[index]=tourslength[index]
                #print veh
                #print veh[0]
                #print veh[0][0]
                a=tourslength
                d=vehlengths
                #print tours
                i=0
                while i<len(tours):
                  tour2=tours[i]
                  #print tour2
                  i+=1
                  if (len(tour2)==2):
                    #print "S"
                    tourslength.remove(a[tours.index(tour2)])
                    vehlengths.remove(d[tours.index(tour2)])
                    tours.remove(tour2)
                    i-=1
                #print "Y"
                e=tours
                #print e
                for l in range(0,t):
                  tour3=tours[l]
                  b=tour3[-1]
                  #print "w"
                  for i in range(0,denum):
                    tours.append([b,i])
                    tourslength.append(self.distances[b][i])
                    vehlengths.append(vehlengths[l])
                #print"x"
              if v>t:
                c=tour[0]
                d=tour[-1]
                for k in range(0,denum):
                  tours.append([d,k])
                  tourslength.append(self.distances[d][k])
                  #print tour
                  vehlengths.append(vehlengths[tours.index(tour)]+tourslength[tours.index(tour)])
                  #print "d",vehlengths[tours.index(tour)],tourslength[tours.index(tour)]
                y=vehlengths[tours.index(tour)]
                for j in range(0,len(tours)):

                  if vehlengths[j]==y:
                    vehlengths[j]+=tourslength[tours.index(tour)]
                    #print "dd"
                for i in range(0,denum):
                  print "d"
                  if [c,i] in tours:
                    print "S"
                    del tourslength[tours.index([c,i])]
                    del vehlengths[tours.index([c,i])]
                    #print vehlengths
                    tours.remove([c,i])
                    print tours
                    print tourslength


                for i in range(0,t):
                  #print veh
                  #print veh[i][-1][-1]
                  if veh[i][-1][-1]==tour[0]:
                    veh[i].append(tour)
                    break

               
            #print tours




            #print tours
        vehlength=[0,0,0,0,0,0]    
        for i in range(0,t):
          for p in range(0,len(veh[i])):
            #print veh[i],veh[i][p],tours.index(veh[i][p])
            #print tourslength[tours.index(veh[i][p])]
            print veh
            vehlength[i]+=tourslength[tours.index(veh[i][p])]
        j=0
        while(j<len(tours)):
          if(len(tours[j]))==2:
            tours.remove(tours[j])
            tourslength.remove(tourslength[j])
          else:
            j+=1
        for l in range(0,t):
          d= veh[l][-1][-1]
          for k in range(0,denum):
                tours.append([d,k])
                veh[l].append([d,k])
                tourslength.append(self.distances[d][k])
        


        return tours, tourslength,veh,vehlength
    """def samedis(self,tours,tourslength):
      c=0.5
      d=0.5
      for tour1 in tours:

        
        
     
        i=1   
        while (i<len(tour1)-1):
            #print(len(tour1))
            #print("!@#!")
            best_dist = self.add(tour1[i-1], tour1[i+1], tour1[i])
            #print("!!!!")
            best_ratio = c*best_dist + d*(tourslength[tours.index(tour1)])
            for tour in tours:
                #print("******")
                if tour != tour1 and len(tour)!=2 :
                 for index in range(len(tour) - 1):
                    dist = self.add(tour[index], tour[index + 1], tour1[i])
                    #print dist
                    ratio = c*dist + d*(tourslength[tours.index(tour)]+tour1[i])
                    if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<150:
                      best_dist = dist
                      new_tour = tour[:index + 1] + [tour1[i]] + tour[index + 1:]
                      tour_index = tours.index(tour)
                      best_ratio = c*best_dist + d*(tourslength[tours.index(tour)])

            if best_ratio != c*best_dist + d*(tourslength[tours.index(tour1)]):
                 tours[tour_index]=new_tour
                 tourslength[tour_index]+= best_dist
                 
                 tourslength[tours.index(tour1)]-=self.add(tour1[i-1], tour1[i + 1], tour1[i])
                 tour1.remove(tour1[i])
            else:
                i+=1             
      

      #print self.distances                #print(i) 
      return tours, tourslength
      """                
    def samedis(self,tours,tourslength,veh,vehlength):
      c=0.2
      d=0.8
      t=3
      for tour1 in tours:
        if len(tour1)!=2:                 
          i=1   
          while (i<len(tour1)-1):
              #print(len(tour1))
              #print("!@#!")
              for j in range(0,t):
                if tour1 in veh[j]:
                  o=j
                  p=veh[j].index(tour1)
                  b=vehlength[j]
              #print veh    #print b,"s" 
              best_dist = self.add(tour1[i-1], tour1[i+1], tour1[i])
              #print("!!!!")
              h=best_dist
              best_ratio = c*best_dist + d*(b)
              for tour in tours:
                  for j in range(0,t):
                    if tour in veh[j]:
                      a=vehlength[j]
                      w=j
                      s=veh[j].index(tour)
                  #print("******")

                  if tour != tour1:
                    #print a
                    for index in range(len(tour) - 1):
                      #print tour
                      #print index
                      dist = self.add(tour[index], tour[index + 1], tour1[i])
                      #print dist
                      if len(tour)==2:
                        y= a+tourslength[tours.index(tour)]+dist
                      else:
                        y=a+dist 
                      ratio = c*dist + d*(y)
                      if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<150:
                        best_dist = dist
                        w1=w
                        s1=s
                        new_tour = tour[:index + 1] + [tour1[i]] + tour[index + 1:]
                        tour_index = tours.index(tour)
                        best_ratio = c*best_dist + d*(y)

              if best_ratio != c*h + d*(b):
                  #print veh
                  tours[tour_index]=new_tour
                  tourslength[tour_index]+= best_dist
                  veh[w1][s1]=new_tour
                  if len(new_tour)==3:
                    vehlength[w1]+=tourslength[tour_index]
                  else:
                    vehlength[w1]+=best_dist

                  
                  #print veh 
                  tourslength[tours.index(tour1)]-=self.add(tour1[i-1], tour1[i + 1], tour1[i])
                  
                  #print o,i
                  #print vehlength[o]
                  vehlength[o]-=self.add(tour1[i-1], tour1[i + 1], tour1[i])
                  print veh
                  print tour1
                  veh[o][p].remove(tour1[i])
                  print tour1
                  print veh
                  if (len(tour1)==2):
                    vehlength[o]-=self.distances[tour1[0]][tour1[1]]
                  #tour1.remove(tour1[i])
                  print veh
              else:
                  i+=1             
      

      #print self.distances                #print(i) 
      return tours, tourslength,veh,vehlength

    def plot (self,tours,color):
      b = ['r','b','g','c']
      j=0      
      for tour in tours:
        if len(tour)!=2:                
          for i in range (0,len(tour)-1):
            if i != len(self.coords)-1:
              plt.plot([self.coords[tour[i]][0],  self.coords[tour[i+1]][0]],[self.coords[tour[i]][1],self.coords[tour[i+1]][1]], color)  
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
tours, lengths,veh,vehlength = x.cheapest_insertion()
print veh
print tours
print lengths
print vehlength
#x.plot(tours)
t=3
b = ['r','b','g','c','m','y']
for i in range(0,t):
  if len(veh[i])!=2:
    x.plot(veh[i],b[i])
plt.show()

tours1, lengths1,veh1,vehlength1 = x.samedis(tours,lengths,veh,vehlength)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,t):
  x.plot(veh1[i],b[i])
plt.show()




