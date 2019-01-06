
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
                a=abs(self.coords[tour[0]][0]-self.coords[city][0])
                b=abs(self.coords[tour[0]][1]-self.coords[city][1])
                if len(tour)!=2:
                  ratio = c*dist+d*(vehlengths[tours.index(tour)]+dist)
                else:
                  ratio = c*dist+d*(vehlengths[tours.index(tour)]+dist+tourslength[tours.index(tour)])
                if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<self.Q:
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

    def nearest_insertion(self,farthest=False):
      denum=8
      t=1
      v=0
      a= len(self.coords)
      #tour = [0,a]
      tours = []
      tourslength = []
      vehlengths=[]
      cantass=[]
      for i in range(0,denum):
        for j in range(0,denum):
          if ([i,j]in tours or [j,i]in tours):
            continue
          else:
            tours.append([i,j])
            cantass.append([])
            tourslength.append(self.distances[i][j])
            vehlengths.append(0)
        # we find the closest node R to the first node
        unass = self.cities[denum:a]
        veh = [[],[],[],[],[],[]]
      ##print unass
      while len(unass) != 0:
          best, best_len,best_ratio = None, 0 if farthest else float('inf'),float('inf')
          t=1
          tour_index = 0
          city1 = 0
          c=0.8
          d=0.2
          #print len(unass)
          vehi=vehindex=None
          # (selection step) given a sub-tour,we find node r not in the
          # sub-tour closest to any node j in the sub-tour,
          # i.e. with minimal c_rj
          for tour in tours:
              for city in unass:
                #print cantass
                #print tour
                if city in cantass[tours.index(tour)]:
                  continue
                # we consider only the distances to nodes already in the tour
                #print tour,city
                _, length = self.closest_neighbor(tour, city, True,farthest)

                if len(tour)!=2:
                  ratio = c*length+d*(vehlengths[tours.index(tour)])

                else:
                  ratio = c*length+d*(vehlengths[tours.index(tour)]+tourslength[tours.index(tour)]+self.add(tour[0],tour[1],city))
                
                if (length > best_len if farthest else ratio < best_ratio):
                    city1, best_len,tour1,best_ratio = city, length,tour,ratio
                    #if len(unass) is 25 or len(unass)is 24:
                      #print best_ratio, tour1,city1

          

          
          ##print city1
          # (insertion step) we find the arc (i, j) in the sub-tour which
          # minimizes cir + crj - cij, and we insert r between i and j

          best_dist, new_tour = float('inf'), None
        
        
          for index in range(len(tour1) - 1):
              dist = self.add(tour1[index], tour1[index + 1], city1)
              ##print dist
              if dist < best_dist and (tourslength[tours.index(tour1)]+dist)<self.Q:
                  best_dist = dist
                  new_tour = tour1[:index + 1] + [city1] + tour1[index + 1:]
                  tour_index = tours.index(tour1)
          if best_dist == float('inf'):
            cantass[tours.index(tour1)].append(city1)
            continue
          for p in range(0,t):
            if tours[tour_index] in veh[p]:            
              vehi=p
              vehindex = veh[p].index(tours[tour_index])
          length, tour, index, city1,vehi,vehindex=best_dist, new_tour, tour_index,city1,vehi,vehindex
          #print tour,index
                  
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
              cantass.append([])
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
                  cantass.append([])
                  tourslength.append(self.distances[b][i])
                  vehlengths.append(vehlengths[l])
              #print"x"
            if v>t:
              c=tour[0]
              d=tour[-1]
              for k in range(0,denum):
                tours.append([d,k])
                cantass.append([])
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
                #print "d"
                if [c,i] in tours:
                  #print "S"
                  del tourslength[tours.index([c,i])]
                  del vehlengths[tours.index([c,i])]
                  #print vehlengths
                  tours.remove([c,i])
                  #print tours
                  #print tourslength
              for i in range(0,t):
                #print veh
                #print veh[i][-1][-1]
                if veh[i][-1][-1]==tour[0]:
                  veh[i].append(tour)
                  break
          #print vehlengths
          print tours

      vehlength=[0,0,0,0,0,0]    
      for i in range(0,t):
        for p in range(0,len(veh[i])):
          #print veh[i],veh[i][p],tours.index(veh[i][p])
          #print tourslength[tours.index(veh[i][p])]
          #print veh
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
        ##print unass
        ##print self.cities
        #return tours,tourslength
                     

    
    def cheapest_insertion(self):
        denum=8
        t=1
        v=0
        a= len(self.coords)
        ##print a
        #tour = [0,a]
        tours = []
        tourslength = []
        vehlengths=[]
        for i in range(0,denum):
          for j in range(0,denum):            
            tours.append([i,j])
            tourslength.append(self.distances[i][j])
            vehlengths.append(0)
        # we find the closest node R to the first node
        unass = self.cities[denum:a]
        veh = [[],[],[],[],[],[]]
        while len(unass) != 0:
            length, tour, index, city1,vehi,vehindex = self.add_closest_to_tour(tours,tourslength,unass,veh,vehlengths)
            ##print unass
            print tours
            #print tourslength
            ##print tour
            ##print veh
            ##print vehi
            ##print vehindex


            if city1 != 0 and tour != 0:
              unass.remove(city1)
              tourslength[index] += length
              tours[index]=tour
            if vehi!=None:
              veh[vehi][vehindex]=tour
              x=vehlengths[tours.index(tour)]
              for j in range(0,len(tours)):
                    
                    if vehlengths[j]==x:
                      ##print"s"
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
                ##print veh
                ##print veh
                
              if v==t:
                veh[v-1].append(tour)
                vehlengths[index]=tourslength[index]
                ##print veh
                ##print veh[0]
                ##print veh[0][0]
                a=tourslength
                d=vehlengths
                ##print tours
                i=0
                while i<len(tours):
                  tour2=tours[i]
                  ##print tour2
                  i+=1
                  if (len(tour2)==2):
                    ##print "S"
                    tourslength.remove(a[tours.index(tour2)])
                    vehlengths.remove(d[tours.index(tour2)])
                    tours.remove(tour2)
                    i-=1
                ##print "Y"
                e=tours
                ##print e
                for l in range(0,t):
                  tour3=tours[l]
                  b=tour3[-1]
                  ##print "w"
                  for i in range(0,denum):
                    tours.append([b,i])
                    tourslength.append(self.distances[b][i])
                    vehlengths.append(vehlengths[l])
                ##print"x"
              if v>t:
                c=tour[0]
                d=tour[-1]
                for k in range(0,denum):
                  tours.append([d,k])
                  tourslength.append(self.distances[d][k])
                  ##print tour
                  vehlengths.append(vehlengths[tours.index(tour)]+tourslength[tours.index(tour)])
                  ##print "d",vehlengths[tours.index(tour)],tourslength[tours.index(tour)]
                y=vehlengths[tours.index(tour)]
                for j in range(0,len(tours)):

                  if vehlengths[j]==y:
                    vehlengths[j]+=tourslength[tours.index(tour)]
                    ##print "dd"
                for i in range(0,denum):
                  #print "d"
                  if [c,i] in tours:
                    #print "S"
                    del tourslength[tours.index([c,i])]
                    del vehlengths[tours.index([c,i])]
                    ##print vehlengths
                    tours.remove([c,i])
                    #print tours
                    #print tourslength


                for i in range(0,t):
                  ##print veh
                  ##print veh[i][-1][-1]
                  if veh[i][-1][-1]==tour[0]:
                    veh[i].append(tour)
                    break

               
            ##print tours




            ##print tours
        vehlength=[0,0,0,0,0,0]    
        for i in range(0,t):
          for p in range(0,len(veh[i])):
            ##print veh[i],veh[i][p],tours.index(veh[i][p])
            ##print tourslength[tours.index(veh[i][p])]
            #print veh
            vehlength[i]+=tourslength[tours.index(veh[i][p])]
        j=0
        while(j<len(tours)):
          if(len(tours[j]))==2:
            tours.remove(tours[j])
            tourslength.remove(tourslength[j])
          else:
            j+=1
        


        return tours, tourslength,veh,vehlength
    """def samedis(self,tours,tourslength):
      c=0.5
      d=0.5
      for tour1 in tours:

        
        
     
        i=1   
        while (i<len(tour1)-1):
            ##print(len(tour1))
            ##print("!@#!")
            best_dist = self.add(tour1[i-1], tour1[i+1], tour1[i])
            ##print("!!!!")
            best_ratio = c*best_dist + d*(tourslength[tours.index(tour1)])
            for tour in tours:
                ##print("******")
                if tour != tour1 and len(tour)!=2 :
                 for index in range(len(tour) - 1):
                    dist = self.add(tour[index], tour[index + 1], tour1[i])
                    ##print dist
                    ratio = c*dist + d*(tourslength[tours.index(tour)]+tour1[i])
                    if ratio < best_ratio and (tourslength[tours.index(tour)]+dist)<self.Q:
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
      

      ##print self.distances                ##print(i) 
      return tours, tourslength
      """
    def farthest_insertion(self,farthest=True):
      denum=8
      t=1
      v=0
      a= len(self.coords)
      #tour = [0,a]
      tours = []
      tourslength = []
      vehlengths=[]
      cantass=[]
      for i in range(0,denum):
        for j in range(0,denum):
          if ([i,j]in tours or [j,i]in tours):
            continue
          else:
            tours.append([i,j])
            cantass.append([])
            tourslength.append(self.distances[i][j])
            vehlengths.append(0)
        # we find the closest node R to the first node
        unass = self.cities[denum:a]
        veh = [[],[],[],[],[],[]]
      ##print unass
      while len(unass) != 0:
          best, best_len,best_ratio = None, 0 if farthest else float('inf'),-float('inf')
          t=1
          tour_index = 0
          city1 = 0
          c=0.8
          d=0.2
          #print len(unass)
          vehi=vehindex=None
          # (selection step) given a sub-tour,we find node r not in the
          # sub-tour closest to any node j in the sub-tour,
          # i.e. with minimal c_rj
          for tour in tours:
              for city in unass:
                #print cantass
                #print tour
                if city in cantass[tours.index(tour)]:
                  continue
                # we consider only the distances to nodes already in the tour
                #print tour,city
                _, length = self.closest_neighbor(tour, city, True,farthest)

                if len(tour)!=2:
                  ratio = c*length-d*(vehlengths[tours.index(tour)])

                else:
                  ratio = c*length-d*(vehlengths[tours.index(tour)]+tourslength[tours.index(tour)]+self.add(tour[0],tour[1],city))
                
                if (ratio > best_ratio if farthest else ratio < best_ratio):
                    city1, best_len,tour1,best_ratio = city, length,tour,ratio
                    #if len(unass) is 25 or len(unass)is 24:
                      #print best_ratio, tour1,city1

          

          
          ##print city1
          # (insertion step) we find the arc (i, j) in the sub-tour which
          # minimizes cir + crj - cij, and we insert r between i and j

          best_dist, new_tour = float('inf'), None
        
        
          for index in range(len(tour1) - 1):
              dist = self.add(tour1[index], tour1[index + 1], city1)
              ##print dist
              if dist < best_dist and (tourslength[tours.index(tour1)]+dist)<self.Q:
                  best_dist = dist
                  new_tour = tour1[:index + 1] + [city1] + tour1[index + 1:]
                  tour_index = tours.index(tour1)
          if best_dist == float('inf'):
            cantass[tours.index(tour1)].append(city1)
            continue
          for p in range(0,t):
            if tours[tour_index] in veh[p]:            
              vehi=p
              vehindex = veh[p].index(tours[tour_index])
          length, tour, index, city1,vehi,vehindex=best_dist, new_tour, tour_index,city1,vehi,vehindex
          #print tour,index
                  
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
              cantass.append([])
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
                  cantass.append([])
                  tourslength.append(self.distances[b][i])
                  vehlengths.append(vehlengths[l])
              #print"x"
            if v>t:
              c=tour[0]
              d=tour[-1]
              for k in range(0,denum):
                tours.append([d,k])
                cantass.append([])
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
                #print "d"
                if [c,i] in tours:
                  #print "S"
                  del tourslength[tours.index([c,i])]
                  del vehlengths[tours.index([c,i])]
                  #print vehlengths
                  tours.remove([c,i])
                  #print tours
                  #print tourslength
              for i in range(0,t):
                #print veh
                #print veh[i][-1][-1]
                if veh[i][-1][-1]==tour[0]:
                  veh[i].append(tour)
                  break
          

      vehlength=[0,0,0,0,0,0]    
      for i in range(0,t):
        for p in range(0,len(veh[i])):
          #print veh[i],veh[i][p],tours.index(veh[i][p])
          #print tourslength[tours.index(veh[i][p])]
          #print veh
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



    def greedy(self):
      denum=8
      c=1
      d=0
      gr=self.grid
      total_len=0
      prevcdep=None
      same_y=[]
      a=len(self.coords)
      tourslength=[]
      tours=[]
      tour=[0]
      tours.append(tour)
      tourslength.append(0)
      unass = self.cities[denum:a]
      f=None
      while len(unass)!=0:
        #print tours
        #print tourslength
        if len(tours[-1])==1:
          x,best_dist=self.closest_city(unass,tours[-1][0])
          #print x
          tours[-1].append(x)
          unass.remove(x)
          tourslength[-1]+=best_dist
        else:
          a=self.coords[tours[-1][-1]][0]
          bb=self.coords[tours[-1][-1]][1]
          best_dist=float('inf')
          city1=None
          for city in unass:
            if self.coords[city][1]==bb:
              same_y.append(city)
          for i in range(0,len(same_y)):
            for  j in range(i+1,len(same_y)):
              if self.distances[same_y[j]][tours[-1][-1]]<self.distances[same_y[i]][tours[-1][-1]]:
                a=same_y[i]
                b=same_y[j]
                same_y[i]=b
                same_y[j]=a
          #print same_y
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
            for i in range(0,denum):
              mindep=abs(self.coords[tours[-1][0]][0]-self.coords[i][0])+abs(self.coords[tours[-1][0]][1]-self.coords[i][1])
              dep=self.distances[city1][i]
              ratio = c*dep+d*mindep          
              if ratio<best_ratio:
                best_dep=dep
                cdep=i
                best_ratio=ratio
            if self.coords[city1][0]==bb and tourslength[-1]>self.Q-self.sqr:
              f=True
            else:
              f= False  
            print f  
            
            if best_dist+tourslength[-1]+best_dep<self.Q:
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
            best_dep=float('inf')
            best_ratio=float('inf')
            cdep=None
            for i in range(0,denum):
              mindep=abs(self.coords[tours[-1][0]][0]-self.coords[i][0])+abs(self.coords[tours[-1][0]][1]-self.coords[i][1])
              dep=self.distances[city1][i]
              ratio = c*dep+d*mindep          
              if ratio<best_ratio:
                best_dep=dep
                cdep=i
                best_ratio=ratio
            if self.coords[city1][0]==bb and tourslength[-1]>self.Q-self.sqr:
              f=True
            else:
              f= False 
                
            print f,self.coords[city1][0]
            if best_dist+tourslength[-1]+best_dep<self.Q:
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
          if len(unass)==0:
            tours[-1].append(prevcdep)
            tourslength[-1]+=prevbest_dep
            tours.append([prevcdep])
            tourslength.append(0)
      xx=[]
      yy=[]
      for i in self.cities:
        xx.append(self.coords[i][0]+dist/2)
        yy.append(self.coords[i][1]+dist/2)

      for i in range(0,len(tourslength)):
        total_len+=tourslength[i]
      return tours,tourslength,total_len,xx,yy




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
      c=['bs','rs','gs','cs','ms','rs','gs','cs','ms']
      for i in range(0,len(self.coords)):
        x.append(self.coords[i][0])
        y.append(self.coords[i][1])
        plt.plot(self.coords[i][0],self.coords[i][1],'rs')

      
      #plt.show()
  
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




 

#r= BaseAlgorithm()

x= TourConstructionHeuristics(dist=dist,grid=20,comm=dist/2,Q=300)

#tours, lengths,veh,vehlength = x.cheapest_insertion()
tours, lengths,total_len,xx,yy = x.greedy()
#tours, lengths,veh,vehlength = x.nearest_insertion()

#print veh
print tours
print lengths
#print total_len
#print vehlength



t=1
b = ['r','b','g','c','m','y']
j=0
#for i in range(0,t):
x.plot(tours)
  #j+=1
plt.show()
k=0
for i in tours:
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
tours1, lengths1,veh1,vehlength1 = x.samedis(tours,lengths,veh,vehlength)

print tours1
print lengths1
print veh1
print vehlength1

b = ['r','b','g','c','m','y']
for i in range(0,t):
  x.plot(veh1[i])

plt.show()


tours,lengths,total=x.greedy()
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

"""
mission planer
qground control
dronecode"""