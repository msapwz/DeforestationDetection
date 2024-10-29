import geopandas as gpd
from pyproj import CRS
from shapely.errors import TopologicalError
from shapely.geometry import Polygon
from tqdm import tqdm
from sentinelsat import SentinelAPI
from sentinelsat.sentinel import SentinelAPIError
import os
import fiona
from requests.exceptions import SSLError
import datetime
import sys
import argparse

fiona.drvsupport.supported_drivers['KML'] = 'rw'

def select_geometries(file,conservation_units,year=2021):
    p  = os.path.dirname(os.path.abspath(__file__))
    locations_file = os.path.join(p,'locations.geojson')
    if not os.path.exists(locations_file):

        gdf = gpd.read_file(file)
        gdf_con = gpd.read_file(conservation_units)

        gdf_year = gdf[gdf['year']==year]

        geom_lenght = []
        areas = []
        id_ = []
        name = []
        geometry = []
        for i in tqdm(gdf_con.index):
            if i==59:
                print("GEOM59")
            try:
                geom = gdf_con['geometry'][i]
                geoms = [i for i in gdf_year['geometry'] if geom.contains(i)]
                a = sum([i.area for i in geoms])
            except TopologicalError:
                print(f"Error with geometry {i}")
            else:    
                if len(geoms)>0:
                    geom_lenght.append(len(geoms))
                    areas.append(a)
                    id_.append(i)
                    name.append(gdf_con['nome'][i])
                    geometry.append(geom)
        gdf = gpd.GeoDataFrame(
            {   
                'id':id_,
                'areas':areas,
                'geoms':geom_lenght,
                'nome':name
            },
            geometry=geometry,
            crs = CRS.from_epsg(4326)
            )

        gdf.sort_values('areas',ascending=False,inplace=True)
        gdf.to_file(locations_file,driver="GeoJSON")
        return gdf
    
    else:
        gdf = gpd.read_file(locations_file)
        return gdf

def download_sentinel2(geometry,
                       d1,
                       d2,
                       outpath,
                       cc,
                        ):

    def get_bounds(geom):
        minx, miny, maxx, maxy = geom.bounds
        p = Polygon([[minx,maxy] , [maxx,maxy] , [maxx,miny] , [minx,miny]])
        return p.wkt

    user = os.environ.get("SENTINELSAT_USERNAME")
    password = os.environ.get("SENTINELSAT_PASSWORD")

    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

    query_kwargs = {
            'area':geometry.wkt, # Try to use the original geometry
            'platformname': 'Sentinel-2',
            'producttype': 'S2MSI2A',
            'date': (d1, d2),
            'cloudcoverpercentage': (0.0,cc)
            }

    try:
        products = api.query(**query_kwargs)
    except (SentinelAPIError,
            ConnectionError,
            SSLError):
        ## Usually this error happen because the geometry is too big (lot of points)
        ## to solve this problem, we use only the bounding box of the geometry
        ## then we will only have 4 points, and the string for requesting 
        ## the api will be small, avoiding errors
        bounds = get_bounds(geometry)
        query_kwargs['area'] = bounds
        products = api.query(**query_kwargs)

    gdf = api.to_geodataframe(products)
    print(f'Images found: {len(gdf)}')

    return gdf

def get_variable(n,name,optional=None):
    try:
        arg = sys.argv[n]
    except IndexError:
        if optional is None:
            raise IndexError(f"You need to pass the variable '{name}' as {n} element.")
        else:
            return optional
    else:
        return arg


def select_dates(geometry,gdf_yd,year):
    try:
        gdf_yd_temp = gdf_yd[gdf_yd.intersects(geometry)]
    except TopologicalError:
        pass
    else:
        gdf_yd_temp = gdf_yd_temp[gdf_yd_temp['year']==year]

        unique = gdf_yd_temp['image_date'].unique()

        unique.sort()

        d1 = datetime.datetime.strptime(unique[0], '%Y-%m-%d') - delta
        d2 = datetime.datetime.strptime(unique[-1], '%Y-%m-%d') + delta

        return (d1,d2)

def get_gdf(geometry,gdf_yd,year,cc):
    dates = select_dates(geometry,gdf_yd,year)
    if dates is not None:
        (d1,d2) = dates
        print("#"*50)
        print(f"Searching for dates {d1.date()} and {d2.date()}")
        print("#" * 50)

        gdf_out = download_sentinel2(
                            geometry,
                            d1,
                            d2,
                            outpath,
                            cc
                        ) 

        gdf_out['tile'] = gdf_out['title'].str[38:44]
        return gdf_out


def download_by_tile(gdf,outpath):
    user = os.environ.get("SENTINELSAT_USERNAME")
    password = os.environ.get("SENTINELSAT_PASSWORD")

    api = SentinelAPI(user, password)#, 'https://scihub.copernicus.eu/dhus')

    uniques = gdf['tile'].unique()

    for u in uniques:
        gdf_temp = gdf[gdf['tile']==u]

        path = os.path.join(outpath,u)
        if not os.path.exists(path):
            os.makedirs(path)

        api.download_all(gdf_temp.index.to_list(),
                         path,
                         n_concurrent_dl=4,
                         lta_retry_delay=1
                         )





if __name__=="__main__":

    parser =  argparse.ArgumentParser(description="Downloading Sentinel 2")
    parser.add_argument("yearly_deforestation",type=str,help="Yearly deforestation file, from TerraBrasilis website.")
    parser.add_argument("conservation_units",type=str,help="Conservation Units file, from TerraBrasilis website.")
    parser.add_argument("outpath",type=str,help="Path where all the files will be downloaded")
    parser.add_argument('-d',"--delta_time",type=int,default = 30,help="Number of day to add as range in the dates to download (more and less X days)")
    parser.add_argument('-g','--geometry',type=int,default=-100,help="Geometry to use to download, if geometry=-100, will download all.")
    parser.add_argument('-c','--cloudcover',type=float,default=20,help="Cloud cover percentage (maximum), from 0 to 100.")

    args = parser.parse_args()

    yearly_deforestation = args.yearly_deforestation
    conservation_units = args.conservation_units
    delta_time = args.delta_time
    outpath = args.outpath
    geometry_n = args.geometry
    cc = args.cloudcover
    #################################################################################################################################
    
    delta = datetime.timedelta(days = delta_time)

    gdf = select_geometries(yearly_deforestation,conservation_units)
    gdf_yd = gpd.read_file(yearly_deforestation)
    
    if geometry_n!=-100:
        gdf = gdf[gdf['id']==geometry_n]
        print(f"The code will download {len(gdf)} geometries")

    for n,i in enumerate(gdf.index):
        for year in [2020,2021]:
            geometry = gdf['geometry'][i]
            gid = gdf['id'][i]

            gdf_out = get_gdf(geometry,gdf_yd,year,cc)
            
            if gdf_out is not None:
                outname = os.path.join(outpath,f'{gid}_{year}.geojson')
                if not os.path.exists(os.path.dirname(outname)):
                    os.makedirs(os.path.dirname(outname))
                
                gdf_out.to_file(outname,driver="GeoJSON")
                download_by_tile(gdf_out,outpath)
            
