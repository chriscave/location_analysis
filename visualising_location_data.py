#This is based on a blog post that can be found at http://beneathdata.com/how-to/visualizing-my-location-history/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import fiona
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from numpy import array
import json
import datetime


shapefilename = '/home/chris/Desktop/cph_shape_files/afstemningsomraade'
shp = fiona.open(shapefilename+'.shp')
coords = shp.bounds
shp.close()
w, h = coords[2] - coords[0], coords[3] - coords[1]
extra = 0.01

figwidth = 14
fig = plt.figure(figsize=(figwidth, figwidth*h/w))
ax = fig.add_subplot(111, facecolor='w', frame_on=False)


m = Basemap(
   projection='tmerc', ellps='WGS84',
    lon_0=np.mean([coords[0], coords[2]]),
    lat_0=np.mean([coords[1], coords[3]]),
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - (extra * h), 
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + (extra * h),
    resolution='h',  suppress_ticks=True)

#50 by 50 square metre of my home location. x_1, x_2, y_1, y_2 should be replace by the map coordinates.
#longpt1, latpt1 = m(x_1,y_1, inverse = True)
#longpt2, latpt2 = m(x_2, y_2, inverse = True)

#opening location data
with open('/home/chris/Desktop/Takeout/Location_History/Location_History.json', 'r') as fh:
 raw = json.loads(fh.read())
 ld = pd.DataFrame(raw['locations'])
 del raw #free up some memory
# convert to typical units
 ld['latitudeE7'] = ld['latitudeE7']/float(1e7) 
 ld['longitudeE7'] = ld['longitudeE7']/float(1e7)
 ld['timestampMs'] = ld['timestampMs'].map(lambda x: float(x)/1000) #to seconds
 ld['datetime'] = ld.timestampMs.map(datetime.datetime.fromtimestamp)
# Rename fields based on the conversions we just did
 ld.rename(columns={'latitudeE7':'latitude', 'longitudeE7':'longitude', 'timestampMs':'timestamp'}, inplace=True)
 ld = ld[ld.accuracy < 1000] #Ignore locations with accuracy estimates over 1000m
 ld = ld[ld.longitude > (coords[0] - (extra * w))]
 ld = ld[ld.longitude < (coords[2] + extra * w)]
 ld = ld[ld.latitude > (coords[1] - (extra * h))]
 ld = ld[ld.latitude < (coords[3] + (extra * h))]
#Removing home location
# ld = ld.drop(ld[(ld.latitude>latpt1) & (ld.latitude < latpt2) & (ld.longitude > longpt1) & (ld.longitude < longpt2)].index)
 ld.reset_index(drop=True, inplace=True)


m.readshapefile(shapefilename, name='cph')

lons_array = np.asarray(list(ld['longitude']))
lats_array = np.asarray(list(ld['latitude']))

x,y = m(lons_array,lats_array)

numhexbins = 50
#plots your location points. I used this to see if the hexbin plot was aligning with the data
#m.plot(x, y,'bo', markersize=0.1)

#plot the hexbin
hx = m.hexbin(x,y,C=None, gridsize=(numhexbins, int(numhexbins*h/w)), bins='log', mincnt=1, edgecolor='none', alpha=1., cmap=plt.get_cmap('Blues'))

#drawing a scale
m.drawmapscale(coords[0] + 0.1, coords[1] + 0.015,
    coords[0], coords[1], 10,
    units='km', barstyle ='fancy',labelstyle='simple',
    fontcolor='#555555',
    zorder=5)


fig.suptitle("My location density in Copenhagen", fontdict={'size':24, 'fontweight':'bold'}, y=0.92)
ax.set_title("Using location data collected from my Android phone via Google Takeout", fontsize=14, y=0.98)
ax.text(1.0, 0.03, "Collected from 2015-2018 \nGeographic data provided by data.kk.dk", 
        ha='right', color='#555555', style='italic', transform=ax.transAxes)
plt.savefig('hexbin_with_home_location.png', dpi=100, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#DEDEDE')

plt.show()

