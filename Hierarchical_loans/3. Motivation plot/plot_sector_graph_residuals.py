import networkx as nx
import EoN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Define Functions 


def plot_graph_residuals():

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

    fig, axs = plt.subplots(5,sharex=True, figsize = (25,5))
    dfsum = pd.DataFrame()
    for a in l1:
        dfsum[a] = df[a]
    axs[0].plot(df['SBI59']-dfsum.sum(axis=1), color = 'skyblue')
    axs[0].set_ylabel('residuals')
    axs[0].set_xlabel('date')
    axs[0].set_title(f'residuals total')

    l1.remove('S15')
    l2.remove([])
    for i in range(4):
        dfsum = pd.DataFrame()
        for a in l2[i]:
            dfsum[a] = df[a]
        axs[i+1].plot(df[f'{l1[i]}']-dfsum.sum(axis=1), color = 'orange')
        axs[i+1].set_ylabel('residuals')
        axs[i+1].set_xlabel('date')
        axs[i+1].set_title(f'residuals {l1[i]}')
    fig.set_tight_layout(True)
    plt.savefig('./residuals_hierarchy', dpi = 400, transparent = True)

if __name__ == '__main__':
    plot_graph_residuals()