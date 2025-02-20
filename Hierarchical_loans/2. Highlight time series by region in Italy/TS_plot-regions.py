import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import geoplot
import geoplot.crs as gcrs
import matplotlib.colors as c
import geopandas as gpd


def plot_TS_regions():
    dataset = pd.read_parquet('./loans_geography.parquet') 
    sns.set_theme(context='notebook', style='whitegrid', palette = 'deep')
    colors = ['skyblue', 'orange', 'limegreen']
    hues = ['Blues', 'Oranges', 'Greens']
    hue_amt = [0.7,0.7,0.5]
    idx = [range(20), range(4), 0]
    for i,region in enumerate(['IT', 'ITC', 'ITC1']):
        
        plt.figure()
        data = gpd.read_file('./limits_IT_regions.geojson')
        data['hue'] = 0
        data['hue'][idx[i]] = hue_amt[i]

        geoplot.choropleth(
            data,
            hue = data['hue'],
            norm = c.Normalize(0,1),
            projection=gcrs.AlbersEqualArea(),
            edgecolor='black',
            cmap = hues[i],
            linewidth=.3,
            figsize=(8, 8)
        )
        plt.savefig(f'./map_{region}', dpi=200, transparent = True)
        
        plt.figure()
        plt.plot(dataset[region], label = region, color = colors[i])
        # plt.plot(dataset['ITC1'], label = 'ITC1',color = 'red')
        plt.xlabel('date')
        plt.ylabel('loan originations')
        plt.legend()
        plt.savefig(f'./TS_{region}', dpi = 200, transparent = True)

if __name__ == '__main__':
    plot_TS_regions()