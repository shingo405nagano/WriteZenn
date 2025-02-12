import os
import random
import shutil
import string
import time
from typing import Any, Dict

from glob import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsProject, QgsVectorLayer
import shapely


def dummy_data(size: int=50_000, lon: float=140.786233, lat: float=40.657981) -> gpd.GeoDataFrame:
    """適当なGeoDataFrameの作成"""
    # 座標の作成。今回は弘前城を中心にランダムな座標を生成する
    lon_list = lon + np.random.normal(0, 0.01, size)
    lat_list = lat + np.random.normal(0, 0.01, size)

    # 適当なコードと年齢を生成
    alphabet = string.ascii_uppercase
    code_list = [
        ''.join(random.choices(alphabet, k=10))
        for _ in range(size)
    ]

    age_list = np.random.normal(45, 15, size).astype(int)

    gdf = gpd.GeoDataFrame(
        data={
            'code': code_list,
            'age': age_list
        },
        geometry=[
            shapely.geometry.Point(lon, lat)
            for lon, lat in zip(lon_list, lat_list)
        ],
        crs='EPSG:4326'
    )
    return gdf



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


################################################################################
################################### GeoPandas ##################################
@stop_watch
def write_file_by_geopandas(
    gdf: gpd.GeoDataFrame, 
    file_path: str, 
    driver: str
) -> Dict[str, Any]: # {'func_result': None, 'elapsed_time': float}
    if driver == 'GPKG':
        gdf.to_file(file_path, driver=driver, layer='test')
    elif driver == 'Parquet':
        gdf.to_parquet(file_path)
    else:
        # KMLは保存の際にする設定が面倒なので、このまま出力
        gdf.to_file(file_path, driver=driver)


@stop_watch
def read_file_by_geopandas(
    file_path: str, 
    driver: str
) -> Dict[str, Any]: # {'func_result': gpd.GeoDataFrame, 'elapsed_time': float}
    if driver == 'GPKG':
        return gpd.read_file(file_path, layer='test')
    elif driver == 'Parquet':
        return gpd.read_parquet(file_path)
    else:
        return gpd.read_file(file_path)


################################### GeoPandas ##################################
################################################################################


################################################################################
##################################### QGIS #####################################
@stop_watch
def read_file_by_qgis(
    file_path: str
) -> Dict[str, Any]: # {'func_result': QgsVectorLayer, 'elapsed_time': float}
    lyr = QgsVectorLayer(file_path, "Read File", "ogr")
    return lyr
    

def measure_rendering_time(
    lyr: QgsVectorLayer
) -> Dict[str, Any]: # {'LyrID': lyr.id(), 'elapsed_time': float}
    renderer = {'elapsed_time': None, 'LyrID': None}
    # レイヤーを作成し、マップキャンバスに追加
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
    driver: str
) -> Dict[str, Any]: # {'func_result': None, 'elapsed_time': float}
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = driver
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
    return None


##################################### QGIS #####################################
################################################################################


################################################################################
##################################### Main #####################################
loop = 1
dir_name = r"C:\Users\makis\Downloads\Mersurements"
file = "OUTPUT."
measurements = {
    'Write-GeoPandas': [],
    'Read-GeoPandas': [],
    'Write-QGIS': [],
    'Renderer-QGIS': [],
    'Read-QGIS': [],
    'FileSize': None
}
contents = {
    'GeoJSON': {
        'result': measurements,
        'fmt': '.geojson'
    },
    'KML': {
        'result': measurements,
        'fmt': '.kml'
    },
    'Esri Shapefile': {
        'result': measurements,
        'fmt': '.shp'
    },
    'FlatGeobuf': {
        'result': measurements,
        'fmt': '.fgb'
    },
    'Parquet': {
        'result': measurements,
        'fmt': '.parquet'
    },
    'GPKG': {
        'result': measurements,
        'fmt': '.gpkg'
    },
}

gdf = dummy_data()

for driver, _item in contents.items():
    fmt = _item.get('fmt')
    for i in range(loop):
        file_path = os.path.join(dir_name, file.replace('.', f'{i}{fmt}'))
        w_time_gpd = write_file_by_geopandas(gdf, file_path, driver).get('elapsed_time')
        r_time_gpd = read_file_by_geopandas(file_path, driver).get('elapsed_time')
        if i == 0:
            file_size = directory_size(dir_name)
        res = read_file_by_qgis(file_path)
        lyr = res.get('func_result')
        r_time_qgis = res.get('elapsed_time')
        rendr_time_qgis = measure_rendering_time(lyr).get('elapsed_time')
        w_time_qgis = write_file_by_qgis(lyr, file_path.replace('.', f'_{i}.'), driver).get('elapsed_time')
        QgsProject.instance().removeMapLayer(lyr.id())
        contents[driver]['result']['Write-GeoPandas'].append(w_time_gpd)
        contents[driver]['result']['Read-GeoPandas'].append(r_time_gpd)
        contents[driver]['result']['Write-QGIS'].append(w_time_qgis)
        contents[driver]['result']['Renderer-QGIS'].append(rendr_time_qgis)
        contents[driver]['result']['Read-QGIS'].append(r_time_qgis)
        contents[driver]['result']['FileSize'] = file_size
    break