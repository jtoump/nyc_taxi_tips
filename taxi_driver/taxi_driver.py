##---------------------------------------------------------------------------------------------------------
# _________  ________     ___    ___ ___          ________  ________  ___  ___      ___ _______   ________     
#|\___   ___\\   __  \   |\  \  /  /|\  \        |\   ___ \|\   __  \|\  \|\  \    /  /|\  ___ \ |\   __  \    
#\|___ \  \_\ \  \|\  \  \ \  \/  / | \  \       \ \  \_|\ \ \  \|\  \ \  \ \  \  /  / | \   __/|\ \  \|\  \   
#     \ \  \ \ \   __  \  \ \    / / \ \  \       \ \  \ \\ \ \   _  _\ \  \ \  \/  / / \ \  \_|/_\ \   _  _\  
#      \ \  \ \ \  \ \  \  /     \/   \ \  \       \ \  \_\\ \ \  \\  \\ \  \ \    / /   \ \  \_|\ \ \  \\  \| 
#       \ \__\ \ \__\ \__\/  /\   \    \ \__\       \ \_______\ \__\\ _\\ \__\ \__/ /     \ \_______\ \__\\ _\ 
#        \|__|  \|__|\|__/__/ /\ __\    \|__|        \|_______|\|__|\|__|\|__|\|__|/       \|_______|\|__|\|__|
#                        |__|/ \|__|                                                                           
#                                                                                                              
##---------------------------------------------------------------------------------------------------------
# Author : Ioannis Toumpalidis
# Last Update : 27 Sep 2020
##---------------------------------------------------------------------------------------------------------
# Collection of scripts to help analyze nyc yellow cab data.
# Link : 
##---------------------------------------------------------------------------------------------------------



import os
import sys

import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString,Point
import numpy as np


import matplotlib.pyplot as plt
import bokeh as bh

from datetime import datetime





class Taxidf():
    
    def __init__(self,src_path=[],taxi_zones_shp_path ="",sample_fraction=None):

        
        self.taxi_data = pd.DataFrame(columns=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                                'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
                                                'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra',
                                                'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
                                                'total_amount', 'pu_day', 'pu_hour', 'do_day', 'do_hour', 'pu_date',
                                                'do_date'])

        if type(src_path)==list :
            for path_ in src_path:
                temp_df=pd.read_csv(path_)
                
                if(sample_fraction is not None):
                    temp_df = temp_df.sample(frac=sample_fraction)
                
                self.taxi_data=self.taxi_data.append(temp_df)
                    

        else:
            
            temp_df=pd.read_csv(src_path)
            if(sample_fraction is not None):
                temp_df = temp_df.sample(frac=sample_fraction)

            self.taxi_data=self.taxi_data.append(temp_df)



        


        self.taxi_zones_shp_path = taxi_zones_shp_path


    def compute_taxi_zones(self):
        """ Create taxi zone geodf
        """

        if(self.taxi_zones_shp_path is not None):
            self.taxi_zones = gpd.read_file(self.taxi_zones_shp_path)
            column_names=["PULocationID","DOLocationID"]

            for col_ in column_names:
                
                grp_=self.taxi_data.groupby(col_,as_index=True)['VendorID'].count()
                count_df=pd.DataFrame({"zone_id":list(grp_.index),col_:grp_})

                self.taxi_zones=pd.merge(self.taxi_zones,count_df,left_on="LocationID",right_on="zone_id")
                del self.taxi_zones["zone_id"]
                
            self.taxi_zones['taxi_demand']=self.taxi_zones['PULocationID']-self.taxi_zones['DOLocationID']
            self.taxi_zones['taxi_activity']=self.taxi_zones['PULocationID']+self.taxi_zones['DOLocationID']
            self.taxi_zones['taxi_demand_class']=self.taxi_zones['taxi_demand'].apply(lambda x: 1 if x>0 else 0)

    #### just grouping functions 
    def group_describe(self,grp_columne=['payment_type'],explained_col='tip_amount'):
        return self.taxi_data.groupby([grp_columne])[explained_col].describe().reset_index()





    def grouped_date(self,column="tpep_pickup_datetime",agg_functions=["count"]):
        grouped_date=self.taxi_data.groupby(["pu_date"])['tpep_pickup_datetime'].agg(agg_functions).reset_index()
        grouped_date = grouped_date.set_index('pu_date')

        return grouped_date

        
    def count_grouped_weekday(self,column="tpep_pickup_datetime",agg_functions=['count']):

        grouped_date=self.grouped_date(column,agg_functions)
        grouped_date['day']=list(pd.Series(grouped_date.index).apply(lambda x: datetime.strptime(x,"%Y-%m-%d").weekday()))
        grouped_day=grouped_date.groupby(["day"])['count'].agg(["mean","max","min","std"]).reset_index()

        return grouped_day

    def grouped_weekday(self,column="tpep_pickup_datetime",agg_functions=['count']):

        grouped_date=self.taxi_data.groupby(["pu_day"])['tpep_pickup_datetime'].agg(agg_functions).reset_index()
        grouped_date = grouped_date.set_index('pu_day')
        return grouped_date



    def grouped_hour(self,column="tpep_pickup_datetime",agg_functions=['count']):

        grouped_date=self.taxi_data.groupby(["pu_hour"])['tpep_pickup_datetime'].agg(agg_functions).reset_index()
        grouped_date = grouped_date.set_index('pu_hour')
        return grouped_date

    
    def convert_dates(self):

        date_time_cols = ["tpep_pickup_datetime","tpep_dropoff_datetime"]

        for col in date_time_cols:
            self.taxi_data[col]=self.taxi_data[col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

        self.taxi_data['pu_day']  = self.taxi_data["tpep_pickup_datetime"].apply(lambda x: x.weekday())
        self.taxi_data['pu_hour'] = self.taxi_data["tpep_pickup_datetime"].apply(lambda x: x.hour)
        self.taxi_data['do_day']  = self.taxi_data["tpep_dropoff_datetime"].apply(lambda x: x.weekday())
        self.taxi_data['do_hour'] = self.taxi_data["tpep_dropoff_datetime"].apply(lambda x: x.hour)
        self.taxi_data['pu_date']= self.taxi_data["tpep_pickup_datetime"].apply(lambda x: str(x.date()))
        self.taxi_data['do_date']= self.taxi_data["tpep_dropoff_datetime"].apply(lambda x: str(x.date())) 
        self.taxi_data['pu_month']=self.taxi_data['tpep_pickup_datetime'].apply(lambda x: x.month)
        self.taxi_data['do_month']=self.taxi_data['tpep_dropoff_datetime'].apply(lambda x: x.month)



    def quantile_plotting(self,axs=None):
        
        if(axs is None):
            fig,axs= plt.subplots(1,2,figsize=(20,20))

        ax=axs[0]

        iqr_zones = taxi_zones[taxi_zones['PULocationID'].between(taxi_zones['PULocationID'].quantile(.25), taxi_zones['PULocationID'].quantile(.75), inclusive=True)]
        iqr_one = taxi_zones[taxi_zones['PULocationID'].between(taxi_zones['PULocationID'].min(), taxi_zones['PULocationID'].quantile(.25), inclusive=True)]
        iqr_last= taxi_zones[taxi_zones['PULocationID'].between(taxi_zones['PULocationID'].quantile(.75), taxi_zones['PULocationID'].max(), inclusive=True)]

        iqr_one.plot(color="purple",ax=ax,label="q1",legend=True)
        iqr_zones.plot(color="green",ax=ax,label="q2",legend=True)
        iqr_last.plot(color="red",ax=ax,label="q3",legend=True)

        ax.set_title("Quantile Plotting - Pickups")

        # for idx,bound_ in enumerate(taxi_zones['geometry']):
        #     geo = bound_.centroid
        #     ax.annotate(text=str(taxi_zones.iloc[idx]['LocationID']),xy=[geo.x, geo.y], color="blue")
            

        iqr_zones = taxi_zones[taxi_zones['DOLocationID'].between(taxi_zones['DOLocationID'].quantile(.25), taxi_zones['DOLocationID'].quantile(.75), inclusive=True)]
        iqr_one = taxi_zones[taxi_zones['DOLocationID'].between(taxi_zones['DOLocationID'].min(), taxi_zones['DOLocationID'].quantile(.25), inclusive=True)]
        iqr_last= taxi_zones[taxi_zones['DOLocationID'].between(taxi_zones['DOLocationID'].quantile(.75), taxi_zones['DOLocationID'].max(), inclusive=True)]
        ax=axs[1]

        iqr_one.plot(color="purple",ax=ax,label="q1",legend=True)
        iqr_zones.plot(color="green",ax=ax,label="q2",legend=True)
        iqr_last.plot(color="red",ax=ax,label="q3",legend=True)

        # for idx,bound_ in enumerate(taxi_zones['geometry']):
        #     geo = bound_.centroid
        #     ax.annotate(text=str(taxi_zones.iloc[idx]['LocationID']),xy=[geo.x, geo.y], color="blue")
        
        ax.set_title("Quantile Plotting - Dropoffs")

        return axs



    def graph_representation(self):


        node_pos = []

        for i in self.taxi_zones['geometry']:
            node_pos.append(list(i.centroid.coords)[0])
                
        node_dict_pos =dict(zip(self.taxi_zones['LocationID'],node_pos))

        G=nx.MultiDiGraph()
        G.add_nodes_from(self.taxi_zones['LocationID'])
        nx.set_node_attributes(G,node_dict_pos,"pos")
        
        self.G_main = G 
        self.pos = node_dict_pos



    def assign_attributes_to_graph(self,od_tip_data=None,attr_list= ['mean',"max","count","std"]):
        

        unique_nodes=list(self.taxi_zones['LocationID'].unique())
        for tupple_od in list(od_tip_data.index):
            
            if((tupple_od[0] in unique_nodes) and (tupple_od[1] in unique_nodes)):
                self.G_main.add_edge(tupple_od[0],tupple_od[1])

        for i in list(self.G_main.edges()):
            dict_ = dict(od_tip_data.loc[i])

            for attr in attr_list:
                self.G_main.edges[(i[0],i[1],0)][attr]=dict_[attr]  
          

    