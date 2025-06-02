
import osmnx as ox

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # or 'Qt5Agg', depending on what's available

def download_data():

    place = 'New York City, New York, USA'
    G = ox.graph_from_place(place, network_type='drive')

    ox.plot_graph(G)
    ox.save_graphml(G, filepath='data/nyc_main_streets.graphml')


def read_data():
    # read from saved file 
    G = ox.load_graphml( filepath='data/nyc_main_streets.graphml')
    fig,ax = plt.subplots(figsize=(10,10))

    ox.plot_graph(G,ax=ax)
    plt.show()



read_data()
