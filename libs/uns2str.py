'''
@brief convert unstructured mesh to structured mesh
@author: hx-w
'''

import trimesh

mesh = trimesh.load_mesh('static/unsmesh/simple_tooth.obj')

class Uns2Str:
    '''
    Use parameterization method
    '''
    def __init__(self) -> None:
        pass
