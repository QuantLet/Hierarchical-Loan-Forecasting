import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import EoN
import geoplot.crs as gcrs
import matplotlib.colors as c
import geoplot
import geopandas as gpd
import seaborn as sns
sns.set_theme(context='notebook', style='whitegrid')

#### Define functions
def plot_graph_geo(savepath = './graphnodes_geography.png', data_path='./loans_geography.parquet'):
    
    #### Read dataset+remove unused features
    Data = pd.read_parquet(data_path)
    cols = list(Data.columns)
    cols.remove('ITHBI12')
    edges = []
    for a in ['ITC', 'ITF', 'ITG', 'ITH', 'ITI'] :
        edges.append(('IT', a))
        for i in range(8):
            if a+f'{i}' in cols: 
                edges.append((a, a+f'{i}'))
    
    #### Build and color graph structure
    G = nx.Graph()
    G.add_nodes_from(cols)
    G.add_edges_from(edges)

    colors = ['skyblue']
    colors.extend(['orange'])
    colors.extend(['green']*4)
    colors.extend(['orange'])
    colors.extend(['green']*6)
    colors.extend(['orange'])
    colors.extend(['green']*2)
    colors.extend(['orange'])
    colors.extend(['green']*3)
    colors.extend(['orange'])
    colors.extend(['green']*4)

    #### Plot and save
    plt.figure(dpi = 200, figsize= (15,5))
    pos = EoN.hierarchy_pos(G,'IT', width= 100)
    options = {
        'node_color': colors,
        'node_size': 1000,
        'width' : 1,
        'edge_color' : 'gray'
    }    
    nx.draw(G, pos=pos, with_labels=True, font_weight = 'bold', **options)
    plt.savefig(savepath, transparent = True)


def plot_graph_sector(savepath = './graphnodes_sectors.png'):
    df = pd.read_parquet('loans_sector.parquet')
    cols = df.columns

    #### Define hierarchy levels
    l0 = ['SBI59']
    l1 = ['S14', 'S13', 'S11', 'S15', 'S12BI7']
    l2 = []
    for a in l1 :
        l2_temp = []
        if a == 'S12BI7':
            for b in cols :
                if 'S12' in b and b!=a :
                    l2_temp.append(b)
        else : 
            for b in cols :
                if a in b and b!=a :
                    l2_temp.append(b)
        
        l2.append(l2_temp)
    l2[0].append('600')
    l2[2].append('450')

    #### Define graph edges
    edges = []
    for i,a in enumerate(l1) :
        edges.append((l0[0],a))
        for b in l2[i] :
            edges.append((a,b))
    cols = l0 + l1 + [a for b in l2 for a in b ]
    colors = ['skyblue']
    colors.extend(['orange']*len(l1))
    colors.extend(['limegreen']*len([a for b in l2 for a in b ]))

    #### Plot and save
    G = nx.Graph()
    G.add_nodes_from(cols)
    G.add_edges_from(edges)
    plt.figure(dpi = 200, figsize= (15,5))
    pos = EoN.hierarchy_pos(G,'SBI59', width= 200)
    options = {
        'node_color': colors,
        'node_size': 1000,
        'font_size': 8,
        'width' : 1,
        'edge_color' : 'gray'
    }    
    nx.draw(G, pos=pos, with_labels=True, font_weight = 'bold', **options)

    plt.savefig(savepath, transparent = True)



if __name__ == '__main__':
    plot_graph_sector()
    plot_graph_geo()