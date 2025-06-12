import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import geoplot
import geoplot.crs as gcrs
import matplotlib.colors as c
import geopandas as gpd


def plot_TS_regions():
    dataset = pd.read_parquet('./loans_full_hierarchy.parquet') 
    sns.set_theme(context='notebook', style='white', palette = 'deep')
    colors = ['C0', 'orange', 'forestgreen']
    hues = ['Blues', 'Oranges', 'Greens']
    hue_amt = [0.7,0.7,0.5]
    idx = [range(20), [0,1,2,6], 0]
    for i,region in enumerate(['IT', 'IT/ITC', 'IT/ITC/ITC1']):
        reg = region.split('/')[-1]
        
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
        plt.savefig(f'./map_{reg}', dpi=200, transparent = True)
        
        plt.figure()
        plt.plot(dataset[region], label = region, color = colors[i])
        # plt.plot(dataset['ITC1'], label = 'ITC1',color = 'red')
        plt.xlabel('date')
        plt.ylabel('loan originations')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol = 1, frameon = False)
        plt.savefig(f'./TS_{reg}', dpi = 200, transparent = True, bbox_inches = 'tight')

if __name__ == '__main__':
    plot_TS_regions()