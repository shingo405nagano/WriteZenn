try:
    import duckdb
except ImportError:
    import subprocess
    subprocess.run('pip install duckdb', shell=True)
    import duckdb

import copy
import os
import random
import shutil
import string
import shutil
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from glob import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import pyproj
from qgis.core import QgsProject
from qgis.core import QgsVectorLayer
from qgis.core import QgsVectorFileWriter
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtCore import QEventLoop
import shapely
duckdb.sql(
"""
INSTALL spatial;
LOAD spatial;
""")



def dummy_data(size: int) -> List[Dict[str, Any]]:
    lon, lat = 140.786233, 40.657981
    lon_list = lon + np.random.normal(0, 0.01, size)
    lat_list = lat + np.random.normal(0, 0.01, size)
    
    # 適当なコードと年齢を生成
    alphabet = string.ascii_uppercase
    code_list = [
        ''.join(random.choices(alphabet, k=10))
        for _ in range(size)
    ]

    age_list = [int(v) for v in np.random.normal(45, 15, size)]

    # geometry を作成
    geometries = [
        shapely.geometry.Point(lon, lat).wkt
        for lon, lat in zip(lon_list, lat_list)
    ]

    datasets = [
        {
            'code': code,
            'age': age,
            'geometry': geometry
        }
        for code, age, geometry in zip(code_list, age_list, geometries)
    ]
    return datasets



def make_path(dirname: str, filename: str, driver: str, layer: str=None):
    file_path = os.path.join(dirname, filename)
    if layer is None:
        return {'file_path': file_path, 'driver': driver}
    return {'file_path': file_path, 'layer': layer, 'driver': driver}


def stop_watch(func):
    """関数の実行時間を計測"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        return {"func_result": result, "elapsed_time": elapsed_time}
    return wrapper
    

def directory_size(dir_name: str) -> float: # MB
    """フォルダ内のファイルサイズを取得。ShapeFileがあるので、フォルダ全体にしている"""
    size = sum([os.path.getsize(file) for file in glob(os.path.join(dir_name, '*'))])
    return round(size / 1048576, 2)
    
    
def delete_contents(folder_path) -> None:
    """フォルダ内のデータを全て削除する"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def make_geodataframe(datasets: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(datasets)
    gdf['geometry'] = gpd.GeoSeries.from_wkt(gdf['geometry'])
    gdf.set_geometry('geometry', inplace=True, crs='EPSG:4326')
    return gdf


################################################################################
################################### GeoPandas ##################################
@stop_watch
def write_geopandas(
    datasets: List[Dict[str, Any]], 
    file_path: str, 
    driver: str,
    layer: Optional[str]=None,
):
    """
    GeoPandas を使ってファイルを書き込む
    """
    gdf = make_geodataframe(datasets)
    if driver == 'Parquet':
        gdf.to_parquet(file_path, index=False)
    elif driver == 'GPKG':
        gdf.to_file(file_path, driver=driver, layer=layer)
    else:
        gdf.to_file(file_path, driver=driver)
    assert os.path.exists(file_path)
    return True


@stop_watch
def read_geopandas(
    file_path: str, 
    driver: str, 
    layer: Optional[str]=None,
):
    """GeoPandas を使ってファイルを読み込む"""
    if driver == 'GPKG':
        _ = gpd.read_file(file_path, layer=layer)
    elif driver == 'Parquet':
        _ = gpd.read_parquet(file_path)
    else:
        _ = gpd.read_file(file_path)
    assert isinstance(_, gpd.GeoDataFrame)
    assert 0 < len(_)
    return True


################################################################################
################################### pyorgio ####################################
@stop_watch
def write_pyogrio(
    datasets: List[Dict[str, Any]], 
    file_path: str, 
    driver: str,
    layer: Optional[str]=None,
    **kwargs
):
    """Pyogrio を使ってファイルを書き込む"""
    if driver == 'Parquet':
        return np.nan
    if layer is None:
        kwargs = {'driver': driver}
    else:
        kwargs = {'driver': driver, 'layer': layer}
    
    pyogrio.write_dataframe(
        make_geodataframe(datasets),
        file_path,
        **kwargs
    )
    assert os.path.exists(file_path)
    return True


@stop_watch
def read_pyogrio(
    file_path: str, 
    driver: str,
    layer: Optional[str]=None,
):
    """Pyogrio を使ってファイルを読み込む"""
    if driver == 'Parquet':
        return np.nan
    if layer is None:
        _ = pyogrio.read_dataframe(file_path)
    else:
        _ = pyogrio.read_dataframe(file_path, layer=layer)
    assert isinstance(_, gpd.GeoDataFrame)
    assert 0 < len(_)
    return True


################################################################################
#################################### DuckDB ####################################
@stop_watch
def read_duckdb(
    file_path: str,
    driver: str,
    layer: Optional[str]=None
):
    """DuckDB を使ってファイルを読み込む"""
    if driver == 'Parquet':
        read_sentence = f"""read_parquet('{file_path}')"""
    elif driver == 'GPKG':
        read_sentence = f"""ST_read('{file_path}', LAYER='{layer}')"""
    else:
        read_sentence = f"""ST_read('{file_path}')"""
    template = \
    """
    SELECT
        * EXCLUDE {geom_column},
        ST_AsText({geom_column}) AS geometry
    FROM
        {read_sentence}
    ;
    """
    try:
        geom_column = 'geom'
        sql = template.format(geom_column=geom_column, read_sentence=read_sentence)
        _df = duckdb.sql(sql).df()
    except:
        geom_column = 'geometry'
        sql = template.format(geom_column=geom_column, read_sentence=read_sentence)
        _df = duckdb.sql(sql).df()
    assert isinstance(_df, pd.DataFrame)
    assert 0 < len(_df)
    gdf = gpd.GeoDataFrame(
        _df.drop(columns=['geometry']),
        geometry=_df['geometry'].apply(shapely.from_wkt),
        crs='EPSG:4326'
    )
    assert isinstance(gdf, gpd.GeoDataFrame)
    return True


################################################################################
##################################### QGIS #####################################
@stop_watch
def read_file_by_qgis(
    file_path: str,
    driver: str,
    layer: Optional[str]=None
) -> Dict[str, Any]: # {'func_result': QgsVectorLayer, 'elapsed_time': float}
    if driver == 'GPKG':
        file_path = f"{file_path}|layername={layer}"
    lyr = QgsVectorLayer(file_path, "Read File", "ogr")
    assert lyr.isValid()
    return lyr
    

def measure_rendering_time(
    file_path: str,
    driver: str,
    layer: Optional[str]=None
) -> Dict[str, Any]: # {'LyrID': lyr.id(), 'elapsed_time': float}
    renderer = {'elapsed_time': None, 'LyrID': None}
    # レイヤーを作成し、マップキャンバスに追加
    result = read_file_by_qgis(file_path, driver, layer)
    lyr = result['func_result']
    renderer['LyrID'] = lyr.id()
    def wait_for_rendering():
        # 描画完了を待つための関数
        nonlocal rendering_time  # elapsed_time を参照
        if iface.mapCanvas().isDrawing():
            QTimer.singleShot(100, wait_for_rendering)  # まだ描画中の場合、再度チェック
        else:
            rendering_time = time.time() - start_time
            renderer['elapsed_time'] = rendering_time
            loop.quit()  # イベントループを終了
    
    # レイヤーの描画時間を計測
    rendering_time = None
    start_time = time.time()
    # レイヤーをマップに追加
    QgsProject.instance().addMapLayer(lyr)
    # 描画完了を待つためのイベントループを作成
    loop = QEventLoop()
    # 描画完了を検出するためのタイマーを開始
    QTimer.singleShot(100, wait_for_rendering)
    # 描画完了までイベントループで待機
    loop.exec_()
    # 経過時間を返す
    return renderer # {'elapsed_time': sec, 'LyrID': lyr.id()}

@stop_watch
def write_file_by_qgis(
    lyr: QgsVectorLayer, 
    file_path: str, 
    driver: str,
    layer: Optional[str]=None
) -> Dict[str, Any]: # {'func_result': None, 'elapsed_time': float}
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = driver
    if driver == 'GPKG':
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
        options.layerName = layer
    error = (
        QgsVectorFileWriter
            .writeAsVectorFormatV2(
                lyr, 
                file_path, 
                QgsProject.instance().transformContext(), 
                options
        )
    )
    assert error[1] != ''
    return True


################################################################################
##################################### Main #####################################
dirname = '..\\datasets\\test_data'

mearsurements = {
    'Write-GeoPandas': [],
    'Read-GeoPandas': [],
    'Write-Pyogrio': [],
    'Read-Pyogrio': [],
    'Read-DuckDB': [],
    'Write-QGIS': [],
    'Read-QGIS': [],
    'Renderer-QGIS': [],
}

size = 5

pack = {
    'GeoJSON': {
        'FilePaths': [
            make_path(dirname, f'test_{i}.geojson', driver='GeoJSON') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
    'KML': {
        'FilePaths': [
            make_path(dirname, f'test_{i}.kml', driver='KML') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
    'Esri Shapefile': {
        'FilePaths': [
            make_path(dirname, f'test_{i}.shp', driver='Esri Shapefile') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
    'FlatGeobuf': {
        'FilePaths': [
            make_path(dirname, f'test_{i}.fgb', driver='FlatGeobuf') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
    'GeoParquet': {
        'FilePaths': [
            make_path(dirname, f'test_{i}.parquet', driver='Parquet') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
    'GeoPackage': {
        'FilePaths': [
            make_path(dirname, 'test.gpkg', driver='GPKG', layer=f'lyr_{i}') 
            for i in range(size)
        ],
        'Result': copy.deepcopy(mearsurements)
    },
}

small_pack = copy.deepcopy(pack)
datasets = dummy_data(size=5_000)
for name, items in pack.items():
    for file_data in items['FilePaths']:
        # Measurement for GeoPandas.
        result = write_geopandas(datasets, **file_data)
        small_pack[name]['Result']['Write-GeoPandas'].append(result['elapsed_time'])
        result = read_geopandas(**file_data)
        small_pack[name]['Result']['Read-GeoPandas'].append(result['elapsed_time'])
        
        # Measurement for Pyogrio.
        result = write_pyogrio(datasets, **file_data)
        small_pack[name]['Result']['Write-Pyogrio'].append(result['elapsed_time'])
        result = read_pyogrio(**file_data)
        small_pack[name]['Result']['Read-Pyogrio'].append(result['elapsed_time'])
        
        # Measurement for DuckDB.
        fmt = os.path.basename(file_data['file_path']).split('.')[1]
        file_path = glob(os.path.join(dirname, f"*.{fmt}"))[0]
        result = read_duckdb(**file_data)
        small_pack[name]['Result']['Read-DuckDB'].append(result['elapsed_time'])

        # Measurement for QGIS.
        result = read_file_by_qgis(**file_data)
        small_pack[name]['Result']['Read-QGIS'].append(result['elapsed_time'])
        result['func_result'] = None
        result = measure_rendering_time(**file_data)
        small_pack[name]['Result']['Renderer-QGIS'].append(result['elapsed_time'])
        



    print(f'{name} is done.')
from pprint import pprint
pprint(small_pack, indent=2)
