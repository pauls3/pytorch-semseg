import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import MITSceneParsingBenchmarkLoader
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader
from ptsemseg.loader.cityscapes_loader_rs19 import cityscapesLoader_rs19
from ptsemseg.loader.railsem19_loader import railsem19Loader
from ptsemseg.loader.railanomalies_loader import railanomaliesLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "cityscapes_rs19": cityscapesLoader_rs19,
        "railsem19": railsem19Loader,
        "railanomalies": railanomaliesLoader
    }[name]
