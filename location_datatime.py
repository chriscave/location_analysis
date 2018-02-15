import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.pyplot import plot_date
import matplotlib.dates as dates
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



freq = 'W' # could also be 'W' (week) or 'D' (day), but month looks nice.
ld = ld.set_index('datetime', drop=False)
ld.index = ld.index.to_period(freq)

ld2016= ld.loc[lambda y: (pd.Timestamp('2016-1-4') < y.datetime) & (y.datetime < pd.Timestamp('2017-1-1')), :]
ld2017= ld.loc[lambda y: (pd.Timestamp('2017-1-1') < y.datetime) & (y.datetime < pd.Timestamp('2018-1-1')), :]
ld2016['week'] = ld2016.datetime.map(lambda x: x.week)
ld2017['week'] = ld2017.datetime.map(lambda x: x.week)


#mindate = ld2016.datetime.min()
#maxdate = ld2016.datetime.max()
per16 = pd.period_range(pd.Timestamp('2016-1-4'), pd.Timestamp('2017-1-1'), freq=freq)
per17 = pd.period_range(pd.Timestamp('2017-1-2'), pd.Timestamp('2017-12-31'), freq=freq)



hm16 = pd.DataFrame(np.zeros([len(per16), 52]) , index=per16)
hm17 = pd.DataFrame(np.zeros([len(per17), 53]) , index=per17)
for period in per16:
    if period in ld2016.index:
       hm16.loc[period] = ld2016.loc[period].week.value_counts() 
for period in per17:
    if period in ld2017.index:
       hm17.loc[period] = ld2017.loc[period].week.value_counts() 

hm17 = hm17.loc[:, '1':'52']

a16 = [list(hm16.iloc[i]) for i in range(0,52)]
a17 = [list(hm17.iloc[i]) for i in range(0,52)]
b16 = np.diag(a16, k =1)
b17 = np.diag(a17)

for i in range(0,51):                        
    hm16.iloc[i] = b16[i]

for i in range(0,52):                        
    hm17.iloc[i] = b17[i]

hm16 = hm16[0]
hm16 = hm16[0:51]
hm17 = hm17[1]


x16 = dates.date2num([p.start_time for p in per16])
x17 = dates.date2num([p.start_time for p in per17])
x16 = x16[0:51]
y16 = list(hm16[0:51])
y17 = list(hm17)

N = 5
cumsum16, moving_aves16 = [0], []

for i, z in enumerate(y16, 1):
    cumsum16.append(cumsum16[i-1] + z)
    if i>=N:
        moving_ave16 = (cumsum16[i] - cumsum16[i-N])/N
        #can do stuff with moving_ave here
        moving_aves16.append(moving_ave16)

cumsum17, moving_aves17 = [0], []

for i, z in enumerate(y17, 1):
    cumsum17.append(cumsum17[i-1] + z)
    if i>=N:
        moving_ave17 = (cumsum17[i] - cumsum17[i-N])/N
        #can do stuff with moving_ave here
        moving_aves17.append(moving_ave17)


w16 = x16[4:52]
w17 = x17[4:53]
plot_date(x16,y16, fmt ='-')
plot_date(x17,y17, fmt ='-')
plot_date(w16, moving_aves16, fmt= '-')
plot_date(w17, moving_aves17, fmt= '-')
